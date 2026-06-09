import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hmmlearn import hmm as hmmlib

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

logger = logging.getLogger(__name__)


def label_historical_windows(df_features: pd.DataFrame) -> list[dict]:
    df = df_features.reset_index(drop=True)
    window_size = CFG["labeling"]["window_size"]
    window_stride = CFG["labeling"]["window_stride"]
    adx_thresh = CFG["labeling"]["rule_adx_trend_threshold"]
    hurst_trend = CFG["labeling"]["rule_hurst_trend_min"]
    hurst_mr = CFG["labeling"]["rule_hurst_mr_max"]
    adf_thresh = CFG["labeling"]["rule_adf_pvalue_max"]
    atr_thresh = CFG["labeling"]["rule_atr_ratio_hv_min"]
    vol_thresh = CFG["labeling"]["rule_vol_ratio_hv_min"]

    # --- Fit HMM ---
    hmm_cols = ["ret_1h", "realized_vol_24h", "atr_14"]
    valid_mask = df[hmm_cols].notna().all(axis=1)
    valid_indices = df.index[valid_mask].tolist()
    X = df.loc[valid_mask, hmm_cols].values

    state_by_row: dict = {}
    if len(X) >= CFG["labeling"]["hmm_n_components"] * 3:
        try:
            model = hmmlib.GaussianHMM(
                n_components=CFG["labeling"]["hmm_n_components"],
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
            model.fit(X)
            raw_states = model.predict(X)
            # Map states: sort by mean realized_vol_24h (index 1)
            state_vol_means = {
                s: np.mean(X[raw_states == s, 1]) if (raw_states == s).any() else 0.0
                for s in range(CFG["labeling"]["hmm_n_components"])
            }
            sorted_states = sorted(state_vol_means, key=state_vol_means.get)
            raw_to_regime = {
                sorted_states[0]: "MEAN_REVERTING",
                sorted_states[1]: "TRENDING",
                sorted_states[2]: "HIGH_VOLATILITY",
            }
            for i, row_idx in enumerate(valid_indices):
                state_by_row[row_idx] = raw_to_regime[raw_states[i]]
        except Exception as e:
            logger.warning(f"HMM fitting failed: {e}")

    n_rows = len(df)
    results: list[dict] = []

    for start in range(0, n_rows - window_size + 1, window_stride):
        end = start + window_size  # exclusive index

        window = df.iloc[start:end]
        mean_adx = float(window["adx_14"].mean())
        mean_hurst = float(window["hurst_exp"].mean())
        mean_adf = float(window["adf_pvalue"].mean())
        mean_bb_width = float(window["bb_width"].mean())
        mean_atr_ratio = float(window["atr_ratio"].mean())
        mean_vol_ratio = float(window["vol_ratio"].mean())
        ema_signal_frac = float((window["ema_9_21_cross"].abs() > 0).mean())
        bb_width_30d_median = float(df.iloc[:end]["bb_width"].median())

        # Primary rule-based label
        if mean_adx > adx_thresh and mean_hurst > hurst_trend and ema_signal_frac > 0.5:
            rule_label = "TRENDING"
        elif mean_hurst < hurst_mr and mean_adf < adf_thresh and mean_bb_width < bb_width_30d_median:
            rule_label = "MEAN_REVERTING"
        elif mean_atr_ratio > atr_thresh and mean_vol_ratio > vol_thresh:
            rule_label = "HIGH_VOLATILITY"
        else:
            trending_score = (
                float(mean_adx > adx_thresh)
                + float(mean_hurst > hurst_trend)
                + float(ema_signal_frac > 0.5)
            ) / 3.0
            mr_score = (
                float(mean_hurst < hurst_mr)
                + float(mean_adf < adf_thresh)
                + float(mean_bb_width < bb_width_30d_median)
            ) / 3.0
            hv_score = (
                float(mean_atr_ratio > atr_thresh) + float(mean_vol_ratio > vol_thresh)
            ) / 2.0

            scores = {
                "TRENDING": trending_score,
                "MEAN_REVERTING": mr_score,
                "HIGH_VOLATILITY": hv_score,
            }
            top_two = sorted(scores.values(), reverse=True)[:2]
            if len(top_two) >= 2 and (top_two[0] - top_two[1]) < 0.15:
                rule_label = "TRANSITIONAL"
            else:
                rule_label = max(scores, key=scores.get)

        # HMM label for last row of window
        last_row_idx = end - 1
        hmm_label = "TRANSITIONAL"
        for candidate in range(last_row_idx, max(last_row_idx - window_size, -1), -1):
            if candidate in state_by_row:
                hmm_label = state_by_row[candidate]
                break

        # Consensus
        if rule_label == hmm_label and rule_label != "TRANSITIONAL":
            label = rule_label
            label_source = "consensus"
        else:
            label = "TRANSITIONAL"
            label_source = "conflict"

        # Feature summary (last row values)
        last_row = window.iloc[-1]
        feature_summary: dict = {}
        for col in df.columns:
            if col == "timestamp":
                feature_summary["timestamp"] = str(last_row["timestamp"])
            elif pd.api.types.is_numeric_dtype(df[col]):
                val = last_row[col]
                feature_summary[col] = float(val) if not pd.isna(val) else 0.0

        # What happened next
        ret_next_4h = None
        ret_next_24h = None
        if end + 4 <= n_rows:
            chunk = df["ret_1h"].iloc[end: end + 4]
            if not chunk.isna().any():
                ret_next_4h = float(chunk.sum())
        if end + 24 <= n_rows:
            chunk24 = df["ret_1h"].iloc[end: end + 24]
            if not chunk24.isna().any():
                ret_next_24h = float(chunk24.sum())

        # regime_duration_candles: consecutive rows after window with same HMM state
        end_state = state_by_row.get(last_row_idx)
        regime_duration = 0
        if end_state is not None:
            for future_idx in range(end, n_rows):
                if state_by_row.get(future_idx) == end_state:
                    regime_duration += 1
                else:
                    break

        ts_col = "timestamp" if "timestamp" in df.columns else None
        ts_start = str(window.iloc[0][ts_col]) if ts_col else str(start)
        ts_end = str(window.iloc[-1][ts_col]) if ts_col else str(end - 1)

        results.append({
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "regime_label": label,
            "label_source": label_source,
            "feature_summary": feature_summary,
            "what_happened_next": {
                "ret_next_4h": ret_next_4h,
                "ret_next_24h": ret_next_24h,
                "regime_persisted": None,
                "regime_duration_candles": regime_duration,
            },
        })

    # Second pass: fill regime_persisted
    for i, w in enumerate(results):
        if i + 1 < len(results):
            w["what_happened_next"]["regime_persisted"] = (
                results[i + 1]["regime_label"] == w["regime_label"]
            )
        else:
            w["what_happened_next"]["regime_persisted"] = False

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_ROOT))
    from data.fetcher import BinanceFetcher
    from data.feature_engine import compute_features

    fetcher = BinanceFetcher()
    df_raw = fetcher.fetch_latest(400)

    rows = []
    for i in range(len(df_raw)):
        window = df_raw.iloc[max(0, i - 199): i + 1]
        if len(window) < 100:
            continue
        try:
            fv = compute_features(window)
            rows.append(vars(fv))
        except Exception as e:
            logger.warning(f"Feature skip at row {i}: {e}")

    df_features = pd.DataFrame(rows)
    print(f"Feature rows: {len(df_features)}")

    windows = label_historical_windows(df_features)
    from collections import Counter
    dist = Counter(w["regime_label"] for w in windows)
    print(f"Label distribution: {dict(dist)}")
    print(f"Consensus windows: {sum(1 for w in windows if w['label_source'] == 'consensus')}")
    print("labeler OK")
