import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

logger = logging.getLogger(__name__)

_DB_PATH = _ROOT / CFG["storage"]["sqlite_path"]


def init_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS regime_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            regime TEXT NOT NULL,
            confidence REAL NOT NULL,
            primary_evidence TEXT,
            contradicting_evidence TEXT,
            reasoning TEXT,
            strategy_implication TEXT,
            analogues_used TEXT,
            hurst_exp REAL,
            adx_14 REAL,
            atr_ratio REAL,
            adf_pvalue REAL,
            raw_llm_response TEXT,
            backend_used TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            from_regime TEXT,
            to_regime TEXT,
            confidence REAL,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_regime(entry: dict) -> None:
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO regime_log (
            timestamp, regime, confidence, primary_evidence, contradicting_evidence,
            reasoning, strategy_implication, analogues_used, hurst_exp, adx_14,
            atr_ratio, adf_pvalue, raw_llm_response, backend_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        entry.get("timestamp"),
        entry.get("regime"),
        entry.get("confidence"),
        json.dumps(entry["primary_evidence"]) if isinstance(entry.get("primary_evidence"), list) else entry.get("primary_evidence"),
        json.dumps(entry["contradicting_evidence"]) if isinstance(entry.get("contradicting_evidence"), list) else entry.get("contradicting_evidence"),
        entry.get("reasoning"),
        entry.get("strategy_implication"),
        json.dumps(entry["analogues_used"]) if isinstance(entry.get("analogues_used"), list) else entry.get("analogues_used"),
        entry.get("hurst_exp"),
        entry.get("adx_14"),
        entry.get("atr_ratio"),
        entry.get("adf_pvalue"),
        entry.get("raw_llm_response"),
        entry.get("backend_used"),
    ))
    conn.commit()
    conn.close()


def log_event(event: dict) -> None:
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO events (timestamp, event_type, from_regime, to_regime, confidence, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        event.get("timestamp"),
        event.get("event_type"),
        event.get("from_regime"),
        event.get("to_regime"),
        event.get("confidence"),
        event.get("notes"),
    ))
    conn.commit()
    conn.close()


def get_recent_logs(n: int = 20) -> list[dict]:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM regime_log ORDER BY id DESC LIMIT ?", (n,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_all_logs_df() -> pd.DataFrame:
    conn = sqlite3.connect(_DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM regime_log ORDER BY id ASC", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


init_db()


if __name__ == "__main__":
    init_db()

    dummy_regime = {
        "timestamp": "2024-01-01T00:00:00",
        "regime": "TRENDING",
        "confidence": 0.85,
        "primary_evidence": ["ADX=32", "Hurst=0.62"],
        "contradicting_evidence": [],
        "reasoning": "Strong trend signals across indicators.",
        "strategy_implication": "Trend-following strategies favored.",
        "analogues_used": ["Analogue 1 matched closely."],
        "hurst_exp": 0.62,
        "adx_14": 32.0,
        "atr_ratio": 1.1,
        "adf_pvalue": 0.12,
        "raw_llm_response": '{"regime":"TRENDING"}',
        "backend_used": "ollama",
    }
    log_regime(dummy_regime)

    dummy_event = {
        "timestamp": "2024-01-01T00:00:00",
        "event_type": "REGIME_CHANGE",
        "from_regime": "MEAN_REVERTING",
        "to_regime": "TRENDING",
        "confidence": 0.85,
        "notes": "Strong breakout detected.",
    }
    log_event(dummy_event)

    print(get_recent_logs(1))
    print("storage OK")
