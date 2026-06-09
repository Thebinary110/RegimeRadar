import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

logger = logging.getLogger(__name__)

CURRENT_STATE: dict = {
    "regime": "UNKNOWN",
    "confidence": 0.0,
    "reasoning": "Agent not yet run.",
    "strategy_implication": "",
    "primary_evidence": [],
    "contradicting_evidence": [],
    "backend_used": "none",
    "last_updated": None,
    "feature_vector_dict": {},
    "cycles_run": 0,
    "last_error": None,
}

SYSTEM_PROMPT: str = """You are a quantitative market regime analyst. Your task is to classify the current market regime for BTC/USDT perpetual futures based on quantitative indicator signals.

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
}"""


def _build_user_prompt(narration: str, analogues: list[dict]) -> str:
    parts = [f"CURRENT MARKET CONDITIONS:\n{narration}\n\n"]
    parts.append("HISTORICAL ANALOGUES FROM SIMILAR MARKET CONDITIONS:\n")
    if analogues:
        for n, analogue in enumerate(analogues, start=1):
            whn = analogue.get("what_happened_next", {})
            ret_4h = whn.get("ret_next_4h")
            ret_24h = whn.get("ret_next_24h")
            parts.append(
                f"[Analogue {n}]\n"
                f"Period: {analogue.get('timestamp_start')} to {analogue.get('timestamp_end')}\n"
                f"Regime: {analogue.get('regime_label')} (source: {analogue.get('label_source')})\n"
                f"Similarity Score: {analogue.get('similarity_score', 0.0):.3f}\n"
                f"What happened next: 4H return = {ret_4h:+.3f}, 24H return = {ret_24h:+.3f}, "
                f"regime persisted = {whn.get('regime_persisted')}, "
                f"duration = {whn.get('regime_duration_candles')} candles\n\n"
                if ret_4h is not None and ret_24h is not None else
                f"[Analogue {n}]\n"
                f"Period: {analogue.get('timestamp_start')} to {analogue.get('timestamp_end')}\n"
                f"Regime: {analogue.get('regime_label')} (source: {analogue.get('label_source')})\n"
                f"Similarity Score: {analogue.get('similarity_score', 0.0):.3f}\n"
                f"What happened next: outcome data unavailable\n\n"
            )
    else:
        parts.append("No historical analogues available (index not built yet).\n")
    return "".join(parts)


def _parse_llm_response(raw: str) -> dict:
    # Attempt 1: direct JSON parse
    try:
        return json.loads(raw.strip())
    except Exception:
        pass

    # Attempt 2: extract first { ... last }
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        pass

    # Attempt 3: regex extract regime
    try:
        m = re.search(r'"regime"\s*:\s*"(\w+)"', raw)
        if m:
            return {
                "regime": m.group(1),
                "confidence": 0.5,
                "reasoning": "Parsed from malformed response",
                "primary_evidence": [],
                "contradicting_evidence": [],
                "analogues_used": [],
                "strategy_implication": "",
            }
    except Exception:
        pass

    # Attempt 4: rule-based fallback (needs fv from caller)
    logger.warning("All JSON parse attempts failed — caller will use rule-based fallback")
    return {}


def _rule_based_regime(fv) -> dict:
    adx_thresh = CFG["labeling"]["rule_adx_trend_threshold"]
    hurst_trend = CFG["labeling"]["rule_hurst_trend_min"]
    hurst_mr = CFG["labeling"]["rule_hurst_mr_max"]
    adf_thresh = CFG["labeling"]["rule_adf_pvalue_max"]
    atr_thresh = CFG["labeling"]["rule_atr_ratio_hv_min"]
    vol_thresh = CFG["labeling"]["rule_vol_ratio_hv_min"]

    evidence = []
    if fv.adx_14 > adx_thresh and fv.hurst_exp > hurst_trend and abs(fv.ema_9_21_cross) == 1:
        regime = "TRENDING"
        if fv.adx_14 > adx_thresh:
            evidence.append(f"ADX={fv.adx_14:.1f} > {adx_thresh}")
        if fv.hurst_exp > hurst_trend:
            evidence.append(f"Hurst={fv.hurst_exp:.3f} > {hurst_trend}")
        if abs(fv.ema_9_21_cross) == 1:
            evidence.append(f"EMA cross={fv.ema_9_21_cross:+.0f}")
    elif fv.hurst_exp < hurst_mr and fv.adf_pvalue < adf_thresh:
        regime = "MEAN_REVERTING"
        evidence.append(f"Hurst={fv.hurst_exp:.3f} < {hurst_mr}")
        evidence.append(f"ADF p={fv.adf_pvalue:.4f} < {adf_thresh}")
    elif fv.atr_ratio > atr_thresh and fv.vol_ratio > vol_thresh:
        regime = "HIGH_VOLATILITY"
        evidence.append(f"ATR ratio={fv.atr_ratio:.2f} > {atr_thresh}")
        evidence.append(f"Vol ratio={fv.vol_ratio:.2f} > {vol_thresh}")
    else:
        regime = "TRANSITIONAL"
        evidence.append(f"No clear dominant regime signal")

    return {
        "regime": regime,
        "confidence": 0.55,
        "reasoning": "Rule-based fallback — LLM unavailable.",
        "primary_evidence": evidence,
        "contradicting_evidence": [],
        "analogues_used": [],
        "strategy_implication": "",
        "backend_used": "rule_based",
    }


def run_cycle() -> None:
    try:
        from data.fetcher import BinanceFetcher
        from data.feature_engine import compute_features
        from agent.narrator import narrate
        from memory.retriever import Retriever
        from agent.llm_client import call, LLMUnavailableError
        from storage.regime_log import log_regime, log_event

        # STEP 1 — OBSERVE
        df = BinanceFetcher().fetch_latest(200)
        if df["timestamp"].iloc[-1] < pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=90):
            logger.warning("Stale data detected, skipping cycle.")
            return
        fv = compute_features(df)

        # STEP 2 — NARRATE
        narration = narrate(fv)

        # STEP 3 — RETRIEVE
        analogues = Retriever().retrieve(narration)

        # STEP 4 — REASON
        user_prompt = _build_user_prompt(narration, analogues)
        result: dict = {}
        try:
            result = call(SYSTEM_PROMPT, user_prompt)
            parsed = _parse_llm_response(result["response_text"])
            if not parsed or "regime" not in parsed:
                parsed = _rule_based_regime(fv)
            else:
                parsed["backend_used"] = result["backend_used"]
        except LLMUnavailableError:
            parsed = _rule_based_regime(fv)

        # STEP 5 — ACT
        entry = {
            "timestamp": fv.timestamp,
            "regime": parsed.get("regime", "UNKNOWN"),
            "confidence": parsed.get("confidence", 0.5),
            "primary_evidence": parsed.get("primary_evidence", []),
            "contradicting_evidence": parsed.get("contradicting_evidence", []),
            "reasoning": parsed.get("reasoning", ""),
            "strategy_implication": parsed.get("strategy_implication", ""),
            "analogues_used": parsed.get("analogues_used", []),
            "hurst_exp": fv.hurst_exp,
            "adx_14": fv.adx_14,
            "atr_ratio": fv.atr_ratio,
            "adf_pvalue": fv.adf_pvalue,
            "raw_llm_response": result.get("response_text", ""),
            "backend_used": parsed.get("backend_used", "rule_based"),
        }
        log_regime(entry)

        previous_regime = CURRENT_STATE["regime"]
        CURRENT_STATE.update({
            "regime": entry["regime"],
            "confidence": entry["confidence"],
            "reasoning": entry["reasoning"],
            "strategy_implication": entry["strategy_implication"],
            "primary_evidence": entry["primary_evidence"],
            "contradicting_evidence": entry["contradicting_evidence"],
            "backend_used": entry["backend_used"],
            "last_updated": entry["timestamp"],
            "feature_vector_dict": vars(fv),
            "cycles_run": CURRENT_STATE["cycles_run"] + 1,
            "last_error": None,
        })

        if (
            previous_regime != entry["regime"]
            and entry["confidence"] >= CFG["agent"]["regime_change_min_confidence"]
        ):
            log_event({
                "timestamp": entry["timestamp"],
                "event_type": "REGIME_CHANGE",
                "from_regime": previous_regime,
                "to_regime": entry["regime"],
                "confidence": entry["confidence"],
                "notes": entry["reasoning"][:200],
            })

        logger.info(
            f"Cycle complete. Regime={entry['regime']} "
            f"Confidence={entry['confidence']:.2f} "
            f"Backend={entry['backend_used']}"
        )

    except Exception as e:
        logger.error(f"run_cycle failed: {e}", exc_info=True)
        CURRENT_STATE["last_error"] = str(e)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(_ROOT))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    run_cycle()
    import pprint
    pprint.pprint(CURRENT_STATE)
    print("react_loop OK")
