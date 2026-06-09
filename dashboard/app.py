import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import subprocess
import time
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st
import yaml

st.set_page_config(layout="wide")

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

from storage.regime_log import get_all_logs_df, get_recent_logs

REGIME_COLOR_MAP = {
    "TRENDING": "#00C853",
    "MEAN_REVERTING": "#2196F3",
    "HIGH_VOLATILITY": "#F44336",
    "TRANSITIONAL": "#9E9E9E",
    "UNKNOWN": "#9E9E9E",
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙ Controls")
refresh_interval = st.sidebar.selectbox("Auto-refresh", ["Manual", "1 min", "5 min"], index=0)

st.sidebar.divider()
if st.sidebar.button("🔄 Rebuild FAISS Index"):
    with st.sidebar.spinner("Building index (this takes several minutes)..."):
        subprocess.run(
            ["python", str(_ROOT / "memory" / "builder.py"), "--rebuild"],
            check=True,
        )
    st.sidebar.success("Index rebuilt.")

st.sidebar.divider()
df_all = get_all_logs_df()
if not df_all.empty and st.sidebar.button("📥 Export Regime Log CSV"):
    st.sidebar.download_button(
        "Download CSV", df_all.to_csv(index=False), "regime_log.csv", "text/csv"
    )

st.sidebar.divider()
st.sidebar.subheader("System Status")

# Ollama status
try:
    resp = requests.get(CFG["ollama"]["base_url"] + "/api/tags", timeout=3)
    resp.raise_for_status()
    st.sidebar.metric("Ollama", "✓ Running")
except Exception:
    st.sidebar.metric("Ollama", "✗ Offline")

# FAISS status
_index_path = _ROOT / CFG["storage"]["faiss_index_path"]
if _index_path.exists():
    try:
        import faiss as _faiss
        _idx = _faiss.read_index(str(_index_path))
        st.sidebar.metric("FAISS Index", f"Loaded ({_idx.ntotal} vectors)")
    except Exception:
        st.sidebar.metric("FAISS Index", "Error loading")
else:
    st.sidebar.metric("FAISS Index", "Not built")

# Last log entry
_recent = get_recent_logs(1)
_last_ts = _recent[0].get("timestamp", "Never") if _recent else "Never"
if _last_ts and _last_ts != "Never":
    _last_ts = _last_ts[-19:]
st.sidebar.metric("Last Log Entry", _last_ts)

# ── ROW 1: STATUS BAR ─────────────────────────────────────────────────────────
latest = get_recent_logs(1)
latest = latest[0] if latest else {}
regime = latest.get("regime", "UNKNOWN")
color = REGIME_COLOR_MAP.get(regime, "#9E9E9E")

cols = st.columns([2, 2, 2, 2, 2])
cols[0].markdown(
    f'<div style="background:{color};padding:10px;border-radius:8px;text-align:center">'
    f'<b style="color:white;font-size:1.4em">{regime}</b></div>',
    unsafe_allow_html=True,
)
cols[1].metric("Confidence", f"{latest.get('confidence', 0):.0%}")
_ts_raw = latest.get("timestamp", "Never")
_ts_disp = _ts_raw[-19:] if _ts_raw and _ts_raw != "Never" else "Never"
cols[2].metric("Last Updated", _ts_disp)
cols[3].metric("Backend", latest.get("backend_used", "—"))
_total = len(df_all) if not df_all.empty else 0
cols[4].metric("Total Cycles", _total)

st.divider()

# ── ROW 2: THREE COLUMNS ──────────────────────────────────────────────────────
left_col, center_col, right_col = st.columns([0.4, 0.3, 0.3])

# LEFT: Regime confidence + indicator time series
with left_col:
    df = get_all_logs_df()
    if not df.empty:
        # Parse timestamps
        try:
            df["ts"] = df["timestamp"].apply(
                lambda x: x[:19] if isinstance(x, str) and len(x) >= 19 else x
            )
            import pandas as pd
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        except Exception:
            import pandas as pd
            df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")

        df100 = df.tail(100).copy()

        # Plot 1: Confidence over time with regime background
        fig1 = go.Figure()
        regimes_series = df100["regime"].tolist()
        ts_series = df100["ts"].tolist()
        conf_series = df100["confidence"].tolist()

        # Add vrects for regime periods
        if len(ts_series) > 1:
            prev_regime = regimes_series[0]
            seg_start = ts_series[0]
            for i in range(1, len(ts_series)):
                if regimes_series[i] != prev_regime or i == len(ts_series) - 1:
                    seg_end = ts_series[i]
                    fig1.add_vrect(
                        x0=seg_start, x1=seg_end,
                        fillcolor=REGIME_COLOR_MAP.get(prev_regime, "#9E9E9E"),
                        opacity=0.15, line_width=0,
                    )
                    prev_regime = regimes_series[i]
                    seg_start = ts_series[i]

        fig1.add_trace(go.Scatter(
            x=ts_series, y=conf_series, mode="lines+markers",
            line={"color": "white", "width": 1.5},
            marker={"size": 4},
            name="Confidence",
        ))
        fig1.update_layout(
            title="Regime Confidence Over Time",
            xaxis_title="Time", yaxis_title="Confidence",
            height=220, margin={"t": 30, "b": 20, "l": 40, "r": 10},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.1)",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Plot 2: ADX, Hurst * 60, ATR Ratio
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ts_series, y=df100["adx_14"].tolist(), name="ADX_14",
            line={"color": "#FF9800"},
        ))
        fig2.add_trace(go.Scatter(
            x=ts_series, y=(df100["hurst_exp"] * 60).tolist(),
            name="Hurst×60", line={"color": "#9C27B0"},
        ))
        fig2.add_trace(go.Scatter(
            x=ts_series, y=df100["atr_ratio"].tolist(), name="ATR Ratio",
            line={"color": "#F44336"},
        ))
        fig2.update_layout(
            title="Key Indicators Over Time",
            height=220, margin={"t": 30, "b": 20, "l": 40, "r": 10},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.1)",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No regime history yet.")

# CENTER: Indicator gauges
with center_col:
    adx_val = float(latest.get("adx_14", 0) or 0)
    hurst_val = float(latest.get("hurst_exp", 0.5) or 0.5)
    atr_val = float(latest.get("atr_ratio", 1.0) or 1.0)
    adf_val = float(latest.get("adf_pvalue", 0.5) or 0.5)

    # ADX Gauge
    fig_adx = go.Figure(go.Indicator(
        mode="gauge+number",
        value=adx_val,
        title={"text": "ADX(14)"},
        gauge={
            "axis": {"range": [0, 60]},
            "threshold": {"line": {"color": "red", "width": 2}, "thickness": 0.75, "value": 25},
            "bar": {"color": color},
        },
    ))
    fig_adx.update_layout(height=200, margin={"t": 30, "b": 0, "l": 20, "r": 20})
    st.plotly_chart(fig_adx, use_container_width=True)

    # Hurst Gauge
    fig_hurst = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hurst_val,
        title={"text": "Hurst Exponent"},
        gauge={
            "axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, 0.42], "color": "lightblue"},
                {"range": [0.42, 0.58], "color": "lightyellow"},
                {"range": [0.58, 1], "color": "lightgreen"},
            ],
            "bar": {"color": color},
        },
    ))
    fig_hurst.update_layout(height=200, margin={"t": 30, "b": 0, "l": 20, "r": 20})
    st.plotly_chart(fig_hurst, use_container_width=True)

    st.metric("ATR Ratio", f"{atr_val:.2f}x", delta=f"{atr_val - 1:.2f}x vs baseline")
    st.metric(
        "ADF p-value",
        f"{adf_val:.4f}",
        delta="stationary ✓" if adf_val < 0.05 else "non-stationary ✗",
    )

# RIGHT: LLM Reasoning Panel
with right_col:
    st.subheader("LLM Reasoning")
    st.info(latest.get("reasoning", "No analysis yet."))

    st.subheader("Primary Evidence")
    _pe_raw = latest.get("primary_evidence", "[]")
    evidence = json.loads(_pe_raw) if isinstance(_pe_raw, str) else (_pe_raw or [])
    for e in evidence:
        st.markdown(f"• {e}")

    st.subheader("Contradicting Signals")
    _ce_raw = latest.get("contradicting_evidence", "[]")
    contra = json.loads(_ce_raw) if isinstance(_ce_raw, str) else (_ce_raw or [])
    if contra:
        for c in contra:
            st.markdown(f"⚠ {c}")
    else:
        st.markdown("_No contradicting signals._")

    st.subheader("Strategy Implication")
    st.success(latest.get("strategy_implication", "—") or "—")

st.divider()

# ── ROW 3: REGIME HISTORY TIMELINE (30 DAYS) ─────────────────────────────────
df_hist = get_all_logs_df()
if df_hist.empty:
    st.info("No regime history yet.")
else:
    try:
        import pandas as pd
        df_hist["ts"] = pd.to_datetime(df_hist["timestamp"].str[:19], errors="coerce")
        cutoff = df_hist["ts"].max() - pd.Timedelta(days=30)
        df30 = df_hist[df_hist["ts"] >= cutoff].copy()

        fig_hist = go.Figure()

        # Add vrects for regime periods
        if len(df30) > 1:
            prev_r = df30["regime"].iloc[0]
            seg_start = df30["ts"].iloc[0]
            mean_conf = [df30["confidence"].iloc[0]]

            for i in range(1, len(df30)):
                cur_r = df30["regime"].iloc[i]
                if cur_r != prev_r or i == len(df30) - 1:
                    seg_end = df30["ts"].iloc[i]
                    fig_hist.add_vrect(
                        x0=seg_start, x1=seg_end,
                        fillcolor=REGIME_COLOR_MAP.get(prev_r, "#9E9E9E"),
                        opacity=float(sum(mean_conf) / len(mean_conf)) * 0.3,
                        line_width=0,
                    )
                    prev_r = cur_r
                    seg_start = df30["ts"].iloc[i]
                    mean_conf = [df30["confidence"].iloc[i]]
                else:
                    mean_conf.append(df30["confidence"].iloc[i])

        fig_hist.add_trace(go.Scatter(
            x=df30["ts"], y=df30["confidence"],
            mode="lines", name="Confidence",
            line={"color": "white", "width": 1.5},
        ))
        fig_hist.update_layout(
            title="Regime History — Last 30 Days",
            xaxis_title="Date", yaxis_title="Confidence",
            height=300, margin={"t": 40, "b": 20, "l": 40, "r": 10},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.1)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    except Exception as ex:
        st.warning(f"Could not render history chart: {ex}")

# ── ROW 4: STATISTICS TABLE ───────────────────────────────────────────────────
st.subheader("Recent Regime Log")
df_display_src = get_all_logs_df()
if not df_display_src.empty:
    import pandas as pd
    _cols = ["timestamp", "regime", "confidence", "adx_14", "hurst_exp",
             "atr_ratio", "adf_pvalue", "backend_used", "reasoning"]
    _available = [c for c in _cols if c in df_display_src.columns]
    df_display = df_display_src.tail(20)[_available].copy()
    if "reasoning" in df_display.columns:
        df_display["reasoning"] = df_display["reasoning"].fillna("").str[:80] + "..."
    if "confidence" in df_display.columns:
        df_display["confidence"] = df_display["confidence"].map(
            lambda x: f"{x:.0%}" if x is not None else "—"
        )
    st.dataframe(df_display, use_container_width=True, hide_index=True)
else:
    st.info("No data yet.")

# ── AUTO-REFRESH ──────────────────────────────────────────────────────────────
if refresh_interval == "1 min":
    time.sleep(60)
    st.rerun()
elif refresh_interval == "5 min":
    time.sleep(300)
    st.rerun()
