import logging
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

logger = logging.getLogger(__name__)


def narrate(fv, timestamp: str = None) -> str:
    from data.feature_engine import FeatureVector

    if isinstance(fv, dict):
        bb_median = fv.get("bb_width_30d_median", None)
        fields = {f: fv.get(f, 0.0) for f in FeatureVector.__dataclass_fields__ if f != "timestamp"}
        fields["timestamp"] = str(fv.get("timestamp", timestamp or "unknown"))
        fv = FeatureVector(**fields)
    else:
        bb_median = None

    ts = timestamp or fv.timestamp

    # ADX interpretation
    if fv.adx_14 > 25:
        adx_interp = "strong trend"
    elif fv.adx_14 >= 20:
        adx_interp = "moderate trend"
    else:
        adx_interp = "weak/no trend"

    # EMA cross interpretation
    if fv.ema_9_21_cross == 1:
        ema_state = "bullish (EMA9 above EMA21)"
    elif fv.ema_9_21_cross == -1:
        ema_state = "bearish (EMA9 below EMA21)"
    else:
        ema_state = "neutral (no clear cross)"

    # Hurst interpretation
    if fv.hurst_exp > 0.6:
        hurst_interp = "strong persistence (trending)"
    elif fv.hurst_exp >= 0.4:
        hurst_interp = "random walk"
    else:
        hurst_interp = "mean-reverting"

    # ADF interpretation
    if fv.adf_pvalue < 0.05:
        adf_interp = "stationary (mean-reverting)"
    else:
        adf_interp = "non-stationary (trending/random)"

    # BB width vs median
    if bb_median is not None:
        bb_vs_median = "expanding" if fv.bb_width > bb_median else "contracting"
    else:
        bb_vs_median = "expanding" if fv.atr_ratio > 1.0 else "contracting"

    return (
        f"Market Snapshot — BTC/USDT Perpetual | {ts} | Timeframe: 1H\n"
        f"\n"
        f"TREND SIGNALS:\n"
        f"- ADX(14) = {fv.adx_14:.1f} → {adx_interp}\n"
        f"- EMA 9/21 crossover: {ema_state}\n"
        f"- Price vs SMA50: {fv.price_vs_sma50:+.2f}%\n"
        f"\n"
        f"MEAN REVERSION SIGNALS:\n"
        f"- Hurst Exponent = {fv.hurst_exp:.3f} → {hurst_interp}\n"
        f"- ADF p-value = {fv.adf_pvalue:.4f} → {adf_interp}\n"
        f"- Z-score vs 20-period mean: {fv.zscore_20:+.2f}σ\n"
        f"- Bollinger Band Width: {fv.bb_width:.4f} ({bb_vs_median} vs 30d median)\n"
        f"\n"
        f"VOLATILITY SIGNALS:\n"
        f"- ATR Ratio vs 30d baseline: {fv.atr_ratio:.2f}x\n"
        f"- Realized Vol Ratio vs 90d baseline: {fv.vol_ratio:.2f}x\n"
        f"- Volume vs 24h mean: {fv.volume_ratio:.2f}x\n"
        f"\n"
        f"RECENT PRICE ACTION:\n"
        f"- 1H return: {fv.ret_1h:+.3f}%\n"
        f"- 4H return: {fv.ret_4h:+.3f}%\n"
        f"- 24H return: {fv.ret_24h:+.3f}%"
    )


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(_ROOT))
    from data.feature_engine import FeatureVector

    dummy = FeatureVector(
        timestamp="2024-06-01 12:00:00+00:00",
        ret_1h=0.002,
        ret_4h=0.008,
        ret_24h=0.015,
        adx_14=32.5,
        di_plus=22.1,
        di_minus=14.3,
        ema_9_21_cross=1.0,
        price_vs_sma50=1.85,
        hurst_exp=0.63,
        adf_pvalue=0.1800,
        zscore_20=0.74,
        bb_width=0.0452,
        atr_14=480.0,
        atr_ratio=1.12,
        realized_vol_24h=0.42,
        vol_ratio=0.95,
        volume_ratio=1.15,
        obv_slope=25000.0,
    )
    print(narrate(dummy))
    print("\nnarrator OK")
