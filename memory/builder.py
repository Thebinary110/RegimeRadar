import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

logger = logging.getLogger(__name__)


def build_index(force_rebuild: bool = False) -> None:
    index_path = _ROOT / CFG["storage"]["faiss_index_path"]
    metadata_path = _ROOT / CFG["storage"]["metadata_path"]

    if index_path.exists() and not force_rebuild:
        print("FAISS index already exists. Pass force_rebuild=True to rebuild.")
        return

    # Step 1 — Fetch historical data
    from data.fetcher import BinanceFetcher
    fetcher = BinanceFetcher()
    days = CFG["binance"]["history_days_memory"]
    logger.info(f"Fetching {days} days of historical data...")
    df = fetcher.fetch_historical(days=days)
    logger.info(f"Fetched {len(df)} candles.")

    # Step 2 — Compute per-row features
    from data.feature_engine import compute_features
    feature_rows = []
    for i in tqdm(range(len(df)), desc="Computing features"):
        window = df.iloc[max(0, i - 199): i + 1]
        if len(window) < 100:
            continue
        try:
            fv = compute_features(window)
            feature_rows.append(vars(fv))
        except Exception as e:
            logger.debug(f"Feature skip at row {i}: {e}")
    if not feature_rows:
        logger.error("No features computed — aborting index build.")
        return

    import pandas as pd
    df_features = pd.DataFrame(feature_rows)

    # Step 3 — Label windows
    from memory.labeler import label_historical_windows
    windows = label_historical_windows(df_features)
    consensus_windows = [w for w in windows if w["label_source"] == "consensus"]
    logger.info(f"Total windows: {len(windows)}, Consensus: {len(consensus_windows)}")

    if not consensus_windows:
        logger.warning("No consensus windows found. Index not built.")
        return

    # Step 3b — Balanced sampling across labels
    from collections import defaultdict
    import random

    by_label: dict = defaultdict(list)
    for w in consensus_windows:
        by_label[w["regime_label"]].append(w)

    per_label_cap: int = CFG.get("faiss", {}).get("per_label_cap", 50)

    sort_keys = {
        "TRENDING": lambda w: w["feature_summary"].get("adx_14", 0),
        "MEAN_REVERTING": lambda w: -w["feature_summary"].get("adf_pvalue", 1),
        "HIGH_VOLATILITY": lambda w: w["feature_summary"].get("atr_ratio", 0),
    }

    balanced_windows: list = []
    for label, wins in by_label.items():
        if label == "TRANSITIONAL":
            continue
        sort_key = sort_keys.get(label, lambda w: 0)
        sorted_wins = sorted(wins, key=sort_key, reverse=True)
        balanced_windows.extend(sorted_wins[:per_label_cap])

    random.shuffle(balanced_windows)
    label_counts = {lb: len(v) for lb, v in by_label.items()}
    logger.info(f"Balanced index: {label_counts} → {len(balanced_windows)} total")
    consensus_windows = balanced_windows

    if not consensus_windows:
        logger.warning("No windows remain after balancing. Index not built.")
        return

    # Step 4 — Load embedding model
    model = SentenceTransformer(CFG["faiss"]["embedding_model"], device="cpu")

    # Step 5 — Embed and build index
    from agent.narrator import narrate
    dim = 384
    index = faiss.IndexHNSWFlat(dim, CFG["faiss"]["hnsw_m"])
    index.hnsw.efConstruction = CFG["faiss"]["ef_construction"]

    embeddings = []
    metadata_records = []

    for i, window in enumerate(tqdm(consensus_windows, desc="Embedding windows")):
        try:
            narration = narrate(window["feature_summary"], window["timestamp_end"])
            emb = model.encode([narration], normalize_embeddings=True)[0]
            embeddings.append(emb)
            metadata_records.append({
                "index_id": i,
                "timestamp_start": window["timestamp_start"],
                "timestamp_end": window["timestamp_end"],
                "regime_label": window["regime_label"],
                "label_source": window["label_source"],
                "feature_summary": window["feature_summary"],
                "what_happened_next": window["what_happened_next"],
            })
        except Exception as e:
            logger.warning(f"Embedding failed for window {i}: {e}")

    if not embeddings:
        logger.error("No embeddings produced — aborting.")
        return

    vectors = np.array(embeddings, dtype=np.float32)
    index.add(vectors)

    # Step 6 — Save
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w") as f:
        for record in metadata_records:
            f.write(json.dumps(record) + "\n")

    logger.info(f"FAISS index built: {len(consensus_windows)} consensus windows indexed.")
    print(f"FAISS index built: {index.ntotal} vectors at {index_path}")


if __name__ == "__main__":
    sys.path.insert(0, str(_ROOT))
    logging.basicConfig(level=logging.INFO)

    force = "--rebuild" in sys.argv
    build_index(force_rebuild=force)

    index_path = _ROOT / CFG["storage"]["faiss_index_path"]
    if index_path.exists():
        idx = faiss.read_index(str(index_path))
        print(f"FAISS index: {idx.ntotal} vectors")

        # Print label distribution from metadata
        import json as _json
        metadata_path = _ROOT / CFG["storage"]["metadata_path"]
        if metadata_path.exists():
            from collections import Counter
            with open(metadata_path) as _f:
                label_dist = Counter(
                    _json.loads(line)["regime_label"]
                    for line in _f if line.strip()
                )
            print(f"Label distribution: {dict(label_dist)}")
    else:
        print("No FAISS index yet (normal on first run before main.py builds it).")
    print("builder OK")
