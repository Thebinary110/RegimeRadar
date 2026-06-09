import json
import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self) -> None:
        index_path = _ROOT / CFG["storage"]["faiss_index_path"]
        metadata_path = _ROOT / CFG["storage"]["metadata_path"]

        if not index_path.exists():
            logger.error(f"FAISS index not found at {index_path}")
            self.available = False
            return

        self.index = faiss.read_index(str(index_path))
        self.index.hnsw.efSearch = CFG["faiss"]["ef_search"]

        with open(metadata_path) as f:
            self.metadata = [json.loads(line) for line in f if line.strip()]

        self.model = SentenceTransformer(CFG["faiss"]["embedding_model"], device="cpu")
        self.available = True
        logger.info(f"Retriever loaded: {self.index.ntotal} vectors.")

    def retrieve(self, narration_text: str, top_k: int = None) -> list[dict]:
        if not self.available:
            return []
        top_k = top_k or CFG["faiss"]["top_k_retrieval"]

        candidate_k = min(top_k * 5, len(self.metadata))
        query_emb = self.model.encode(
            [narration_text], normalize_embeddings=True
        ).astype(np.float32)

        distances, indices = self.index.search(query_emb, candidate_k)

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            record = self.metadata[idx].copy()
            record["similarity_score"] = float(1 - dist)
            record["_embedding_idx"] = int(idx)
            candidates.append(record)

        if not candidates:
            return []

        lambda_mmr: float = CFG["faiss"].get("mmr_lambda", 0.6)

        candidate_embs = np.array(
            [self.index.reconstruct(c["_embedding_idx"]) for c in candidates],
            dtype=np.float32,
        )

        selected: list[dict] = []
        selected_embs: list[np.ndarray] = []
        remaining = list(range(len(candidates)))

        for _ in range(top_k):
            if not remaining:
                break

            if not selected:
                best_idx = max(remaining, key=lambda i: candidates[i]["similarity_score"])
            else:
                best_score = -float("inf")
                best_idx = remaining[0]
                selected_embs_arr = np.array(selected_embs, dtype=np.float32)
                for i in remaining:
                    sim_to_query = candidates[i]["similarity_score"]
                    sims_to_selected = candidate_embs[i] @ selected_embs_arr.T
                    max_sim_to_selected = float(np.max(sims_to_selected))
                    mmr_score = (
                        lambda_mmr * sim_to_query
                        - (1 - lambda_mmr) * max_sim_to_selected
                    )
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i

            selected.append(candidates[best_idx])
            selected_embs.append(candidate_embs[best_idx])
            remaining.remove(best_idx)

        for s in selected:
            s.pop("_embedding_idx", None)

        return selected


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_ROOT))
    r = Retriever()
    if r.available:
        results = r.retrieve("BTC trending strongly with ADX 35", top_k=2)
        for res in results:
            print(res.get("regime_label"), res.get("similarity_score"))
        print("retriever OK")
    else:
        print("retriever unavailable (no index yet)")
        print("retriever OK")
