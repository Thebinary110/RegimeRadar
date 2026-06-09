import logging
import time
from pathlib import Path

import ccxt
import pandas as pd
import yaml
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

logger = logging.getLogger(__name__)


class BinanceFetcher:
    def __init__(self) -> None:
        api_key = CFG["binance"]["api_key"]
        api_secret = CFG["binance"]["api_secret"]
        params = {
            'defaultType': 'future',
            'fetchCurrencies': False,
            'options': {
                'fetchCurrencies': False,
                'adjustForTimeDifference': True,
            }
        }
        if api_key and api_secret:
            params["apiKey"] = api_key
            params["secret"] = api_secret
        self.exchange = ccxt.binanceusdm(params)
        self.symbol: str = CFG["binance"]["symbol"]
        self.timeframe: str = CFG["binance"]["timeframe"]
    def _fetch_with_retry(self, since_ms: int, limit: int) -> list:
        last_exc = None
        for attempt in range(1, 4):
            try:
                return self.exchange.fetch_ohlcv(
                    self.symbol, self.timeframe, since=since_ms, limit=limit
                )
            except (ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
                last_exc = e
                wait = 2 ** attempt
                logger.warning(f"Fetch retry {attempt}/3 after {wait}s: {e}")
                time.sleep(wait)
        logger.error(f"All retries exhausted: {last_exc}")
        raise last_exc

    def fetch_historical(self, days: int) -> pd.DataFrame:
        ms_per_candle = 3600 * 1000
        total_candles = days * 24
        since_ms = self.exchange.milliseconds() - total_candles * ms_per_candle
        all_rows: list = []
        chunk = 1000
        pbar = tqdm(total=total_candles, desc="Fetching historical candles", unit="candles")
        while since_ms < self.exchange.milliseconds() - ms_per_candle:
            rows = self._fetch_with_retry(since_ms, chunk)
            if not rows:
                break
            all_rows.extend(rows)
            pbar.update(len(rows))
            since_ms = rows[-1][0] + ms_per_candle
            if len(rows) < chunk:
                break
        pbar.close()
        return self._to_df(all_rows)

    def fetch_latest(self, n_candles: int = 200) -> pd.DataFrame:
        ms_per_candle = 3600 * 1000
        since_ms = self.exchange.milliseconds() - n_candles * ms_per_candle
        rows = self._fetch_with_retry(since_ms, n_candles)
        return self._to_df(rows)

    @staticmethod
    def _to_df(rows: list) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        return df


if __name__ == "__main__":
    fetcher = BinanceFetcher()
    df = fetcher.fetch_latest(10)
    print(df.iloc[-1])
    print("fetcher OK")
