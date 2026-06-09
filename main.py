import logging
import sys
import time
from pathlib import Path

import requests
import yaml

_ROOT = Path(__file__).parent
with open(_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

# Configure logging (file + stream)
_LOG_PATH = _ROOT / CFG["storage"]["log_path"]
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logger = logging.getLogger(__name__)

# Create required directories
for _d in ["storage", "memory", "logs", "data"]:
    (_ROOT / _d).mkdir(parents=True, exist_ok=True)

# Initialize DB
from storage.regime_log import init_db
init_db()

# --- Startup checks ---

# 1. Check Ollama
try:
    r = requests.get(CFG["ollama"]["base_url"] + "/api/tags", timeout=5)
    r.raise_for_status()
    print("✓ Ollama running")
except Exception:
    print("⚠ Ollama not running — will use Groq fallback")

# 2. Check Groq
if CFG["groq"]["api_key"]:
    print("✓ Groq API key configured")
else:
    print("⚠ Groq API key not set — rule-based fallback only if Ollama also fails")

# 3. Check Binance
from data.fetcher import BinanceFetcher
try:
    BinanceFetcher().fetch_latest(5)
    print("✓ Binance connected")
except Exception as e:
    print(f"✗ Binance connection failed — cannot continue. ({e})")
    sys.exit(1)

# 4. Check / Build FAISS index
from memory.builder import build_index
_index_path = _ROOT / CFG["storage"]["faiss_index_path"]
if _index_path.exists():
    print(f"✓ FAISS index found")
else:
    print("FAISS index not found. Building from 2 years of historical data...")
    print("This will take several minutes. Do not interrupt.")
    build_index(force_rebuild=False)
    print("✓ FAISS index built successfully")

# 5. Immediate cycle
from agent.react_loop import run_cycle, CURRENT_STATE
print("Running initial regime detection cycle...")
run_cycle()
print(
    f"✓ Initial cycle complete. Regime: {CURRENT_STATE['regime']} "
    f"({CURRENT_STATE['confidence']:.0%} confidence)"
)

# 6. Scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

scheduler = BackgroundScheduler()
scheduler.add_job(
    run_cycle,
    CronTrigger(minute=CFG["binance"]["poll_offset_minutes"]),
)
scheduler.start()
print(f"✓ Scheduler running. Next cycle at :{CFG['binance']['poll_offset_minutes']:02d} each hour.")

# 7. Dashboard instructions
print("\n" + "=" * 50)
print("Dashboard ready. Run in a separate terminal:")
print(f"  streamlit run {_ROOT / 'dashboard' / 'app.py'}")
print("=" * 50 + "\n")

# 8. Keep alive
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\nShutting down...")
    scheduler.shutdown()
