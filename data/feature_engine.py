import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    timestamp: str
    ret_1h: float
    ret_4h: float
    ret_24h: float
    adx_14: float
    di_plus: float
    di_minus: float
    ema_9_21_cross: float
    price_vs_sma50: float
    hurst_exp: float
    adf_pvalue: float
    zscore_20: float
    bb_width: float
    atr_14: float
    atr_ratio: float
    realized_vol_24h: float
    vol_ratio: float
    volume_ratio: float
    obv_slope: float


def compute_hurst(prices: np.ndarray) -> float:
    try:
        log_returns = np.diff(np.log(prices))
        if len(log_returns) < 20:
            return 0.5
        max_lag = min(100, len(log_returns) // 2)
        lags = [n for n in [10, 20, 40, 80, max_lag] if n <= len(log_returns) // 2 and n >= 2]
        lags = sorted(set(lags))
        if len(lags) < 2:
            return 0.5
        rs_values = []
        for n in lags:
            n_subs = len(log_returns) // n
            if n_subs < 1:
                rs_values.append(np.nan)
                continue
            rs_list = []
            for i in range(n_subs):
                sub = log_returns[i * n:(i + 1) * n]
                mean_adj = sub - np.mean(sub)
                cumsum = np.cumsum(mean_adj)
                R = np.max(cumsum) - np.min(cumsum)
                S = np.std(sub, ddof=1)
                if S > 0 and R > 0:
                    rs_list.append(R / S)
            rs_values.append(np.mean(rs_list) if rs_list else np.nan)
        valid = [
            (np.log(n), np.log(rs))
            for n, rs in zip(lags, rs_values)
            if not np.isnan(rs) and rs > 0
        ]
        if len(valid) < 2:
            return 0.5
        log_ns, log_rs = zip(*valid)
        slope = float(np.polyfit(log_ns, log_rs, 1)[0])
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5


def _nan_fallback(series: pd.Series, field_name: str) -> float:
    val = series.iloc[-1]
    if pd.isna(val):
        fallback = series.expanding().mean().iloc[-1]
        logger.debug(f"NaN in {field_name}, using expanding mean: {fallback}")
        return float(fallback) if not pd.isna(fallback) else 0.0
    return float(val)


def compute_features(df: pd.DataFrame) -> FeatureVector:
    if len(df) < 100:
        raise ValueError("Insufficient candle history")

    close = df["close"].astype(float).reset_index(drop=True)
    high = df["high"].astype(float).reset_index(drop=True)
    low = df["low"].astype(float).reset_index(drop=True)
    volume = df["volume"].astype(float).reset_index(drop=True)

    # Returns
    ret_1h = _nan_fallback(np.log(close / close.shift(1)), "ret_1h")
    ret_4h = _nan_fallback(np.log(close / close.shift(4)), "ret_4h")
    ret_24h = _nan_fallback(np.log(close / close.shift(24)), "ret_24h")

    # ADX / DI
    adx_df = ta.adx(high, low, close, length=14)
    adx_14 = _nan_fallback(adx_df[f"ADX_14"], "adx_14")
    di_plus = _nan_fallback(adx_df[f"DMP_14"], "di_plus")
    di_minus = _nan_fallback(adx_df[f"DMN_14"], "di_minus")

    # EMA cross
    ema9 = ta.ema(close, length=9)
    ema21 = ta.ema(close, length=21)
    e9 = float(ema9.iloc[-1]) if not pd.isna(ema9.iloc[-1]) else float(close.iloc[-1])
    e21 = float(ema21.iloc[-1]) if not pd.isna(ema21.iloc[-1]) else float(close.iloc[-1])
    if e9 > e21 * 1.001:
        ema_9_21_cross = 1.0
    elif e9 < e21 * 0.999:
        ema_9_21_cross = -1.0
    else:
        ema_9_21_cross = 0.0

    # SMA50
    sma50 = ta.sma(close, length=50)
    sma50_val = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else float(close.expanding().mean().iloc[-1])
    price_vs_sma50 = (float(close.iloc[-1]) - sma50_val) / sma50_val * 100

    # Hurst
    hurst_exp = compute_hurst(close.values[-100:])

    # ADF
    try:
        adf_pvalue = float(adfuller(close.values[-100:], autolag="AIC")[1])
    except Exception:
        adf_pvalue = 0.5

    # Zscore
    roll_mean = close.rolling(20).mean()
    roll_std = close.rolling(20).std()
    zscore_series = (close - roll_mean) / roll_std
    zscore_20 = _nan_fallback(zscore_series, "zscore_20")

    # Bollinger Band width
    bb = ta.bbands(close, length=20, std=2.0)
    bbl_col = [c for c in bb.columns if c.startswith("BBL")][0]
    bbm_col = [c for c in bb.columns if c.startswith("BBM")][0]
    bbu_col = [c for c in bb.columns if c.startswith("BBU")][0]
    bb_width_series = (bb[bbu_col] - bb[bbl_col]) / bb[bbm_col]
    bb_width = _nan_fallback(bb_width_series, "bb_width")

    # ATR & ATR ratio
    atr_series = ta.atr(high, low, close, length=14)
    atr_14 = _nan_fallback(atr_series, "atr_14")
    atr_mean_long = atr_series.rolling(30 * 24).mean().iloc[-1]
    if pd.isna(atr_mean_long):
        atr_mean_long = atr_series.expanding().mean().iloc[-1]
        logger.debug("NaN in atr_ratio denominator, using expanding mean")
    atr_ratio = (atr_series.iloc[-1] / atr_mean_long) if (atr_mean_long and not pd.isna(atr_mean_long)) else 1.0

    # Realized vol
    log_ret_1h_series = np.log(close / close.shift(1))
    realized_vol_24h = float(np.std(log_ret_1h_series.values[-24:]) * np.sqrt(8760))
    rv_series = log_ret_1h_series.rolling(24).std() * np.sqrt(8760)
    rv_mean_long = rv_series.rolling(90 * 24).mean().iloc[-1]
    if pd.isna(rv_mean_long):
        rv_mean_long = rv_series.expanding().mean().iloc[-1]
        logger.debug("NaN in vol_ratio denominator, using expanding mean")
    vol_ratio = (realized_vol_24h / float(rv_mean_long)) if (rv_mean_long and not pd.isna(rv_mean_long)) else 1.0

    # Volume ratio
    vol_mean_24h = volume.rolling(24).mean().iloc[-1]
    volume_ratio = (float(volume.iloc[-1]) / float(vol_mean_24h)) if (vol_mean_24h and not pd.isna(vol_mean_24h)) else 1.0

    # OBV slope
    obv_changes = np.sign(log_ret_1h_series.fillna(0)) * volume
    obv = obv_changes.cumsum()
    obv_last10 = obv.values[-10:].astype(float)
    if len(obv_last10) >= 2 and not np.any(np.isnan(obv_last10)):
        slope, *_ = linregress(np.arange(len(obv_last10)), obv_last10)
        obv_slope = float(slope)
    else:
        obv_slope = 0.0

    return FeatureVector(
        timestamp=str(df["timestamp"].iloc[-1]),
        ret_1h=ret_1h,
        ret_4h=ret_4h,
        ret_24h=ret_24h,
        adx_14=adx_14,
        di_plus=di_plus,
        di_minus=di_minus,
        ema_9_21_cross=ema_9_21_cross,
        price_vs_sma50=price_vs_sma50,
        hurst_exp=hurst_exp,
        adf_pvalue=adf_pvalue,
        zscore_20=zscore_20,
        bb_width=bb_width,
        atr_14=atr_14,
        atr_ratio=atr_ratio,
        realized_vol_24h=realized_vol_24h,
        vol_ratio=vol_ratio,
        volume_ratio=volume_ratio,
        obv_slope=obv_slope,
    )


if __name__ == "__main__":
    from data.fetcher import BinanceFetcher
    fetcher = BinanceFetcher()
    df = fetcher.fetch_latest(200)
    fv = compute_features(df)
    for field, val in vars(fv).items():
        print(f"  {field}: {val}")
    print("feature_engine OK")
