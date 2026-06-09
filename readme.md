# Project 1 — LLM-Powered Regime Detection Agent
## Architecture Specification & Build Contract

---

## 1. Project Purpose

A production-grade autonomous agent that continuously monitors BTC/USDT perpetual futures on Binance, detects the current market regime (Trending / Mean-Reverting / High-Volatility), and explains its reasoning using a locally-running quantized LLM. The system combines classical quantitative indicators, a FAISS+HNSW vector memory of historically-labeled regime windows, and a ReAct (Reason + Act) agent loop to produce explainable, regime-aware signal outputs. All results are visualized on a live Streamlit dashboard.

---

## 2. Regime Taxonomy

The system detects exactly three regimes:

| Regime | Definition | Primary Signals |
|---|---|---|
| `TRENDING` | Sustained directional price movement | ADX > 25, Hurst > 0.58, EMA crossover aligned |
| `MEAN_REVERTING` | Price oscillates around a stable mean | Hurst < 0.42, ADF rejects unit root (p < 0.05), BB width contracting |
| `HIGH_VOLATILITY` | No directional conviction, elevated realized vol | ATR ratio > 1.5x baseline, BB width expanding, volume spike |

A fourth label `TRANSITIONAL` is assigned when signals conflict across regimes (e.g. ADX > 25 but Hurst < 0.42). This is stored in memory but not acted on — it indicates the system is between regimes.

---

## 3. System Architecture — Module Map

```
project1_regime_agent/
├── README.md                  ← this file (source of truth)
├── requirements.txt
├── config.yaml                ← all tuneable parameters live here
├── main.py                    ← entry point, starts agent loop + dashboard
│
├── data/
│   ├── fetcher.py             ← ccxt Binance perpetual OHLCV pull
│   └── feature_engine.py     ← all indicator computation, outputs FeatureVector
│
├── memory/
│   ├── labeler.py             ← hybrid HMM + rule-based regime labeling for historical data
│   ├── builder.py             ← embeds narrated windows, builds FAISS HNSW index
│   └── retriever.py           ← similarity search, returns top-K analogues with metadata
│
├── agent/
│   ├── narrator.py            ← FeatureVector → structured natural language string
│   ├── llm_client.py          ← Ollama HTTP client wrapper (Qwen2.5-3B Q4)
│   └── react_loop.py          ← full ReAct orchestration: Observe→Narrate→Retrieve→Reason→Act
│
├── storage/
│   └── regime_log.py          ← SQLite logger for all regime decisions + LLM outputs
│
└── dashboard/
    └── app.py                 ← Streamlit dashboard, reads from SQLite + live agent state
```

---

## 4. Data Pipeline

### 4.1 Source
- Exchange: Binance, market: `BTC/USDT` perpetual futures (`BTC/USDT:USDT`)
- Library: `ccxt` (async mode for live polling, sync for historical bulk pull)
- Timeframe: `1h` candles
- Historical depth: 90 days for live feature computation window; 730 days (2 years) for FAISS memory construction

### 4.2 Feature Vector Specification
Every 1h candle close triggers computation of the following `FeatureVector` (a Python dataclass):

**Returns:**
- `ret_1h`: 1-period log return
- `ret_4h`: 4-period rolling log return
- `ret_24h`: 24-period rolling log return

**Trend Indicators:**
- `adx_14`: Average Directional Index, period 14
- `di_plus`: +DI component
- `di_minus`: -DI component
- `ema_9_21_cross`: +1 if EMA9 > EMA21, -1 if below, 0 if within 0.1% band
- `price_vs_sma50`: (close - SMA50) / SMA50, as percentage

**Mean-Reversion Indicators:**
- `hurst_exp`: Hurst exponent computed via R/S analysis over last 100 candles (rolling)
- `adf_pvalue`: ADF test p-value on last 100 candle close prices
- `zscore_20`: z-score of current close vs 20-period rolling mean/std
- `bb_width`: (BB upper - BB lower) / BB middle, period 20, 2 std

**Volatility Indicators:**
- `atr_14`: Average True Range, period 14
- `atr_ratio`: atr_14 / (30-day rolling mean of atr_14)
- `realized_vol_24h`: std of last 24 log returns, annualized
- `vol_ratio`: realized_vol_24h / (90-day rolling mean of realized_vol_24h)

**Volume Indicators:**
- `volume_ratio`: current volume / 24-period rolling mean volume
- `obv_slope`: slope of OBV over last 10 candles (linear regression coefficient)

All values are stored as floats. NaN values from insufficient history are filled with rolling expanding mean during warm-up period.

---

## 5. Regime Labeling for Historical Memory (Hybrid Method)

Historical windows are labeled using **both** methods and merged:

### 5.1 Rule-Based Labeling (Primary Signal)
Applied per 100-candle window with 50-candle stride:

```
IF adx_14 > 25 AND hurst_exp > 0.55 AND abs(ema_9_21_cross) == 1:
    label = TRENDING
ELIF hurst_exp < 0.45 AND adf_pvalue < 0.05 AND bb_width < bb_width_30d_median:
    label = MEAN_REVERTING
ELIF atr_ratio > 1.4 AND vol_ratio > 1.3:
    label = HIGH_VOLATILITY
ELIF signal_conflict (any two regimes score within 15% of each other):
    label = TRANSITIONAL
ELSE:
    label = most_dominant_single_signal
```

### 5.2 HMM Labeling (Validation Layer)
- 3-state Gaussian HMM fitted on `[ret_1h, realized_vol_24h, atr_14]` using `hmmlearn`
- States are post-hoc mapped to regime labels by matching state means to regime definitions
- Used to cross-validate rule-based labels: if HMM and rule-based disagree, window is labeled `TRANSITIONAL` regardless

### 5.3 Label Quality Filter
Only windows where rule-based and HMM agree are stored in FAISS as high-confidence memories. Disagreement windows are stored separately in SQLite for analysis but excluded from retrieval.

---

## 6. FAISS HNSW Memory Layer

### 6.1 Embedding
- Model: `sentence-transformers/all-MiniLM-L6-v2` (CPU, 80MB, 384-dim output)
- Input to embedding: the **narrated text** of the feature vector (not the raw numbers)
- Each stored vector = embedding of the narrated window description

### 6.2 Index Configuration
- Index type: `faiss.IndexHNSWFlat(384, 32)` — HNSW with M=32 (connectivity parameter)
- HNSW `efConstruction`: 200 (high quality index build)
- HNSW `efSearch`: 50 (quality at query time)
- Distance metric: L2 (cosine similarity approximated via normalized embeddings)

Why HNSW over flat IVF: HNSW gives O(log n) search with no training step required, and at our index size (~5,000 windows from 2 years of 1H data with 50-candle stride) it outperforms IVF on both speed and recall.

### 6.3 Stored Metadata (Parallel to FAISS Index)
Each vector's corresponding metadata is stored in a JSON-lines file indexed by position:
```json
{
  "index_id": 0,
  "timestamp_start": "2023-01-01T00:00:00Z",
  "timestamp_end": "2023-05-15T03:00:00Z",
  "regime_label": "TRENDING",
  "label_source": "consensus",
  "feature_summary": { ... },
  "what_happened_next": {
    "ret_next_4h": 0.023,
    "ret_next_24h": 0.041,
    "regime_persisted": true,
    "regime_duration_candles": 34
  }
}
```

### 6.4 Retrieval
At inference: embed current narration → HNSW search → return top-3 matches with full metadata. The `what_happened_next` field is included in the LLM context so the model can reason about historical outcomes of similar setups.

---

## 7. ReAct Agent Loop

The loop runs on a scheduler aligned to hourly candle close (triggered at :02 past each hour to allow candle settlement).

### Step 1 — OBSERVE
- Pull latest 200 candles from Binance via ccxt
- Compute FeatureVector for current window
- Check data freshness: if latest candle > 90 minutes old, log WARNING and skip cycle

### Step 2 — NARRATE
Convert FeatureVector to structured English via `narrator.py`. Output format:

```
Market Snapshot — BTC/USDT Perpetual | {timestamp} | Timeframe: 1H

TREND SIGNALS:
- ADX(14) = {adx_14:.1f} → {interpretation: "strong trend" if >25 else "weak/no trend"}
- EMA 9/21 crossover: {state}
- Price vs SMA50: {price_vs_sma50:+.2f}%

MEAN REVERSION SIGNALS:
- Hurst Exponent = {hurst_exp:.3f} → {interpretation}
- ADF p-value = {adf_pvalue:.4f} → {interpretation}
- Z-score vs 20-period mean: {zscore_20:+.2f}σ
- Bollinger Band Width: {bb_width:.4f} ({bb_width_vs_median} vs 30d median)

VOLATILITY SIGNALS:
- ATR Ratio vs 30d baseline: {atr_ratio:.2f}x
- Realized Vol Ratio vs 90d baseline: {vol_ratio:.2f}x
- Volume vs 24h mean: {volume_ratio:.2f}x

RECENT PRICE ACTION:
- 1H return: {ret_1h:+.3f}%
- 4H return: {ret_4h:+.3f}%
- 24H return: {ret_24h:+.3f}%
```

### Step 3 — RETRIEVE
- Embed current narration with `all-MiniLM-L6-v2`
- Query FAISS HNSW index for top-3 most similar historical windows
- Format retrieved analogues as: `[Analogue 1: {timestamp}, Regime: {label}, Similarity: {score:.3f}, What happened next: {outcome}]`

### Step 4 — REASON (LLM Call)
Single call to Ollama (`qwen2.5:3b-instruct-q4_K_M`).

**System Prompt:**
```
You are a quantitative market regime analyst. Your task is to classify the current market regime for BTC/USDT perpetual futures based on quantitative indicator signals.

The three possible regimes are:
- TRENDING: Sustained directional price movement. Characterized by ADX > 25, Hurst > 0.55, aligned EMA crossover.
- MEAN_REVERTING: Price oscillating around a stable mean. Characterized by Hurst < 0.45, ADF p < 0.05, contracting Bollinger Bands.
- HIGH_VOLATILITY: Elevated realized volatility with no directional conviction. Characterized by ATR ratio > 1.4x and vol ratio > 1.3x baseline.

You will be given:
1. Current market indicators (narrated)
2. Three historical analogues from similar past market conditions with their outcomes

Respond ONLY in the following JSON format, no preamble, no markdown:
{
  "regime": "TRENDING" | "MEAN_REVERTING" | "HIGH_VOLATILITY" | "TRANSITIONAL",
  "confidence": 0.0-1.0,
  "primary_evidence": ["list of 2-3 specific indicator values that most support this call"],
  "contradicting_evidence": ["list of any signals that contradict the call, empty list if none"],
  "analogues_used": ["brief note on which historical analogue was most relevant and why"],
  "reasoning": "2-3 sentence explanation of the regime call in plain English",
  "strategy_implication": "one sentence on what strategy class is appropriate for this regime"
}
```

**User Prompt:** Current narration + formatted analogues

**LLM Parameters:** temperature=0.1, top_p=0.9, max_tokens=512

### Step 5 — ACT
- Parse JSON response (with fallback regex extraction if JSON malformed)
- Validate: regime must be one of four valid labels, confidence must be 0-1 float
- Write to SQLite `regime_log` table
- Update in-memory shared state dict (read by dashboard)
- If regime changed from previous cycle AND confidence > 0.7: write `REGIME_CHANGE` event to SQLite `events` table
- Log full cycle to `agent.log`

---

## 8. Storage — SQLite Schema

### Table: `regime_log`
```sql
CREATE TABLE regime_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    regime TEXT NOT NULL,
    confidence REAL NOT NULL,
    primary_evidence TEXT,       -- JSON array stored as string
    contradicting_evidence TEXT, -- JSON array stored as string
    reasoning TEXT,
    strategy_implication TEXT,
    analogues_used TEXT,
    hurst_exp REAL,
    adx_14 REAL,
    atr_ratio REAL,
    adf_pvalue REAL,
    raw_llm_response TEXT,
    backend_used TEXT    -- "ollama" | "groq_llama" | "groq_mixtral" | "rule_based"
);
```

### Table: `events`
```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,    -- REGIME_CHANGE | CONFIDENCE_DROP | DATA_WARN
    from_regime TEXT,
    to_regime TEXT,
    confidence REAL,
    notes TEXT
);
```

---

## 9. Configuration — config.yaml

```yaml
binance:
  symbol: "BTC/USDT:USDT"
  timeframe: "1h"
  history_days_live: 90
  history_days_memory: 730
  candles_per_window: 100
  poll_offset_minutes: 2  # minutes past hour to poll

ollama:
  base_url: "http://localhost:11434"
  model: "qwen2.5:3b-instruct-q4_K_M"
  temperature: 0.1
  top_p: 0.9
  max_tokens: 512
  timeout_seconds: 60

faiss:
  hnsw_m: 32
  ef_construction: 200
  ef_search: 50
  top_k_retrieval: 3
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

labeling:
  hmm_n_components: 3
  rule_adx_trend_threshold: 25
  rule_hurst_trend_min: 0.55
  rule_hurst_mr_max: 0.45
  rule_adf_pvalue_max: 0.05
  rule_atr_ratio_hv_min: 1.4
  rule_vol_ratio_hv_min: 1.3
  window_size: 100
  window_stride: 50

agent:
  confidence_change_threshold: 0.7
  regime_change_min_confidence: 0.65

storage:
  sqlite_path: "storage/regime_agent.db"
  faiss_index_path: "memory/regime_hnsw.index"
  metadata_path: "memory/regime_metadata.jsonl"
  log_path: "logs/agent.log"
```

---

## 10. Streamlit Dashboard Specification

### Layout: Wide mode, 3-column structure

**Row 1 — Live Status Bar (full width)**
- Current regime badge (color-coded: green=TRENDING, blue=MEAN_REVERTING, red=HIGH_VOLATILITY, grey=TRANSITIONAL)
- Confidence meter (progress bar, 0-100%)
- Last updated timestamp
- Cycles run counter

**Row 2 — Left Column (40% width): Price Chart**
- Plotly candlestick chart, last 100 candles
- Background shading by regime period (colored bands)
- EMA9 and EMA21 overlaid as lines
- Bollinger Bands overlaid
- Regime change events marked as vertical dashed lines

**Row 2 — Center Column (30% width): Indicator Gauges**
- ADX gauge (0-60 scale, threshold line at 25)
- Hurst Exponent gauge (0-1 scale, threshold bands at 0.42 and 0.58)
- ATR Ratio bar (baseline at 1.0x, warning at 1.4x)
- ADF p-value display with pass/fail indicator

**Row 2 — Right Column (30% width): LLM Reasoning Panel**
- Current regime reasoning text (from LLM)
- Primary evidence bullets
- Contradicting evidence bullets (if any)
- Strategy implication text
- Most relevant historical analogue summary

**Row 3 — Full Width: Regime History Timeline**
- Plotly timeline chart showing regime over last 30 days
- Color-coded by regime
- Overlaid with BTC/USDT price (secondary y-axis)
- Confidence as opacity of regime band

**Row 4 — Full Width: Regime Statistics Table**
- Last 20 regime log entries as dataframe
- Columns: Timestamp, Regime, Confidence, ADX, Hurst, ATR Ratio, Reasoning (truncated)

**Sidebar:**
- Refresh interval selector (1min / 5min / manual)
- Rebuild FAISS index button (with progress indicator)
- Export regime log as CSV button
- System status: Ollama connection, Binance connection, FAISS index size

**Auto-refresh:** `st.rerun()` on timer, configurable interval

---

## 11. Execution Entry Point — main.py

```
main.py behavior:
1. Load config.yaml
2. Initialize SQLite (create tables if not exist)
3. Check Ollama connection (GET /api/tags) — exit with clear error if not running
4. Check Binance connectivity via ccxt
5. Check if FAISS index exists at config path:
   - If YES: load existing index and metadata
   - If NO: run full historical data pull + labeling + index build (logs progress)
6. Start ReAct agent loop in background thread (APScheduler, cron: "2 * * * *")
7. Run one immediate agent cycle on startup
8. Launch Streamlit dashboard via subprocess OR print instructions to run separately
```

---

## 12. Error Handling Contract

- **Binance API rate limit:** Exponential backoff, max 3 retries, then skip cycle and log
- **Ollama timeout:** If LLM call exceeds 60s, log WARNING and use rule-based fallback (no LLM reasoning, regime from rules only, confidence capped at 0.6)
- **Malformed LLM JSON:** Attempt regex extraction of regime field; if fails, use rule-based fallback
- **FAISS index missing at runtime:** Log ERROR, disable retrieval step, run agent without analogues (degrades gracefully)
- **Insufficient candle history (< 100 candles):** Skip cycle, log WARNING, retry next cycle
- **NaN in feature vector:** Fill with expanding rolling mean for numeric stability; log which features were filled

---

## 13. Dependencies — requirements.txt

```
ccxt>=4.2.0
pandas>=2.0.0
numpy>=1.24.0
pandas-ta>=0.3.14b
statsmodels>=0.14.0
hmmlearn>=0.3.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
requests>=2.31.0
openai>=1.30.0
streamlit>=1.32.0
plotly>=5.18.0
pyyaml>=6.0
apscheduler>=3.10.0
sqlalchemy>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
tqdm>=4.66.0
```

---

## 14. API Keys & Credentials

### Binance
- Uses authenticated private REST endpoints via ccxt for higher rate limits and access to perpetual futures OHLCV
- Keys stored in `config.yaml` under `binance.api_key` and `binance.api_secret`
- Permissions required: **Read Only** — no trading permissions needed
- If keys are absent or empty, ccxt falls back to public endpoints (lower rate limits, same data)

### Ollama (Primary LLM)
- Runs fully locally, no API key required
- Model: `qwen2.5:3b-instruct-q4_K_M`
- Must be running via `ollama serve` before starting the agent

### Groq (Fallback LLM)
- Free tier, no payment required — sign up at console.groq.com
- Primary fallback model: `llama-3.1-8b-instant` (fastest, free tier)
- Secondary fallback model: `mixtral-8x7b-32768` (stronger reasoning, also free tier)
- Key stored in `config.yaml` under `groq.api_key`
- Groq API is OpenAI-compatible — use `openai` Python client pointed at `https://api.groq.com/openai/v1`
- Fallback trigger: Ollama timeout (>60s) OR Ollama connection refused OR Ollama returns empty response

### Fallback Chain (in order)
```
1. Ollama local (qwen2.5:3b-instruct-q4_K_M)  ← primary, always tried first
2. Groq llama-3.1-8b-instant                   ← fallback if Ollama fails
3. Groq mixtral-8x7b-32768                     ← fallback if llama-3.1 fails
4. Rule-based regime detection only             ← last resort, no LLM reasoning, confidence capped at 0.55
```

The `llm_client.py` module must implement this chain transparently — `react_loop.py` calls a single `call()` function and never needs to know which backend was used. The response includes a `backend_used` field so the dashboard can display which LLM produced the current reasoning.

---

## 15. Updated config.yaml (replaces Section 9)

```yaml
binance:
  symbol: "BTC/USDT:USDT"
  timeframe: "1h"
  history_days_live: 90
  history_days_memory: 730
  candles_per_window: 100
  poll_offset_minutes: 2
  api_key: ""        # paste Binance Read-Only API key here
  api_secret: ""     # paste Binance API secret here

ollama:
  base_url: "http://localhost:11434"
  model: "qwen2.5:3b-instruct-q4_K_M"
  temperature: 0.1
  top_p: 0.9
  max_tokens: 512
  timeout_seconds: 60

groq:
  api_key: ""        # paste Groq free tier API key here
  base_url: "https://api.groq.com/openai/v1"
  primary_model: "llama-3.1-8b-instant"
  fallback_model: "mixtral-8x7b-32768"
  temperature: 0.1
  max_tokens: 512
  timeout_seconds: 30

faiss:
  hnsw_m: 32
  ef_construction: 200
  ef_search: 50
  top_k_retrieval: 3
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

labeling:
  hmm_n_components: 3
  rule_adx_trend_threshold: 25
  rule_hurst_trend_min: 0.55
  rule_hurst_mr_max: 0.45
  rule_adf_pvalue_max: 0.05
  rule_atr_ratio_hv_min: 1.4
  rule_vol_ratio_hv_min: 1.3
  window_size: 100
  window_stride: 50

agent:
  confidence_change_threshold: 0.7
  regime_change_min_confidence: 0.65

storage:
  sqlite_path: "storage/regime_agent.db"
  faiss_index_path: "memory/regime_hnsw.index"
  metadata_path: "memory/regime_metadata.jsonl"
  log_path: "logs/agent.log"
```

---

## 16. Build Constraints

1. **Binance API key optional but recommended** — falls back to public endpoints gracefully if keys empty
2. **No paid AI API keys** — Ollama is free/local, Groq free tier only
3. **No GPU required at runtime** — sentence-transformers runs on CPU, Ollama handles GPU/CPU automatically
4. **Single-machine deployment** — everything runs locally, no Docker required
5. **All file paths relative to project root** — no hardcoded absolute paths, use pathlib.Path throughout
6. **Config-driven** — all thresholds and parameters in `config.yaml`, never hardcoded in source
7. **Graceful degradation** — system runs through the full fallback chain before giving up
8. **Idempotent index build** — running builder.py twice does not duplicate entries
9. **Dashboard reads only from SQLite + shared state** — never calls Binance or any LLM directly
10. **Backend transparency** — every regime log entry records which LLM backend produced it