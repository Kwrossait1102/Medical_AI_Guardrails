from __future__ import annotations
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

DEFAULT_EMBED_MODEL = os.getenv("SCOPE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BLOCK_THRESHOLD = float(os.getenv("SCOPE_BLOCK_THRESHOLD", "0.68"))
WHITELIST_MARGIN = float(os.getenv("SCOPE_WHITELIST_MARGIN", "0.08"))
DEFAULT_PROTO_PATH = os.getenv("SCOPE_PROTO_PATH", None)
CUSTOM_SEEDS_JSON = os.getenv("SCOPE_SEEDS_JSON", None)

RESTRICTED_KEYS = ("diagnosis", "report", "dosage", "plan")
ALL_KEYS = (*RESTRICTED_KEYS, "whitelist")


@dataclass
class ProtoSet:
    vectors: Dict[str, np.ndarray]
    texts: Dict[str, List[str]]


class OutputPrivacyGuard:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        proto_path: Optional[str] = DEFAULT_PROTO_PATH,
        block_threshold: float = BLOCK_THRESHOLD,
        whitelist_margin: float = WHITELIST_MARGIN,
        seeds_json: Optional[str] = CUSTOM_SEEDS_JSON,
        forbid_fallback: bool = False,
        model = None
    ):
        self.model_name = model_name
        self.block_threshold = block_threshold
        self.whitelist_margin = whitelist_margin
        self.model = model or SentenceTransformer(self.model_name)

        if proto_path and os.path.exists(proto_path):
            try:
                if not _HAS_TORCH:
                    raise RuntimeError("Torch not installed; cannot load .pt")
                self.protos = self._load_prototypes_pt(proto_path)
                return
            except Exception as e:
                if forbid_fallback:
                    raise RuntimeError(f"Failed to load prototypes from {proto_path}") from e
        elif proto_path and not os.path.exists(proto_path):
            if forbid_fallback:
                raise FileNotFoundError(f"Prototype file not found: {proto_path}")

        seeds = self._load_seed_texts_from_json(seeds_json) if seeds_json else self._default_seed_texts()
        self.protos = self._build_prototypes(seeds)

        if proto_path:
            if not _HAS_TORCH:
                raise RuntimeError("Torch not installed; cannot save .pt")
            os.makedirs(os.path.dirname(proto_path), exist_ok=True)
            self._save_prototypes_pt(proto_path, self.protos)

    def check(self, text: str) -> Tuple[bool, Optional[str]]:
        if not text or not text.strip():
            return True, None
        sims = self._similarities(text)
        restricted_max = max(sims[k] for k in RESTRICTED_KEYS)
        if sims["whitelist"] >= restricted_max + self.whitelist_margin:
            return True, None
        if restricted_max >= self.block_threshold:
            return False, "Personalized medical advice detected."
        return True, None

    def check_debug(self, text: str) -> Tuple[bool, Optional[str], Dict[str, float]]:
        is_safe, reason = self.check(text)
        sims = self._similarities(text)
        return is_safe, reason, sims

    def _build_prototypes(self, seed_texts: Dict[str, List[str]]) -> ProtoSet:
        dim = self._emb_dim()
        vecs: Dict[str, np.ndarray] = {}

        for cls, samples in seed_texts.items():
            if not samples:
                vecs[cls] = np.zeros((dim,), dtype=np.float32)
                continue
            emb = self._encode(samples)
            proto = self._l2_normalize(emb.mean(axis=0))
            vecs[cls] = proto

        for k in ALL_KEYS:
            if k not in vecs:
                vecs[k] = np.zeros((dim,), dtype=np.float32)
            seed_texts.setdefault(k, [])

        return ProtoSet(vectors=vecs, texts=seed_texts)

    def _save_prototypes_pt(self, path: str, protos: ProtoSet) -> None:
        payload = {
            "vectors": {k: v for k, v in protos.vectors.items()},
            "texts": protos.texts,
            "meta": {
                "model_name": self.model_name,
                "block_threshold": self.block_threshold,
                "whitelist_margin": self.whitelist_margin,
            },
        }
        torch.save(payload, path)

    def _load_prototypes_pt(self, path: str) -> ProtoSet:
        data = torch.load(path, map_location="cpu", weights_only=False)
        vectors = {k: np.asarray(v, dtype=np.float32) for k, v in data.get("vectors", {}).items()}
        texts = {k: list(v) for k, v in data.get("texts", {}).items()}

        dim = self._emb_dim()
        for k in ALL_KEYS:
            if k not in vectors:
                vectors[k] = np.zeros((dim,), dtype=np.float32)
            if k not in texts:
                texts[k] = []

        meta = data.get("meta", {})
        self.model_name = meta.get("model_name", self.model_name)
        self.block_threshold = float(meta.get("block_threshold", self.block_threshold))
        self.whitelist_margin = float(meta.get("whitelist_margin", self.whitelist_margin))
        return ProtoSet(vectors=vectors, texts=texts)

    def _similarities(self, text: str) -> Dict[str, float]:
        q = self._encode([text])[0]
        sims = {cls: float(np.dot(q, vec)) for cls, vec in self.protos.vectors.items()}
        for k in ALL_KEYS:
            sims.setdefault(k, -1.0)
        return sims

    def _encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128
        )
        return np.asarray(embs, dtype=np.float32)

    def _emb_dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-12
        return v / n

    def _default_seed_texts(self) -> Dict[str, List[str]]:
        return {
            "diagnosis": [
                "You are likely suffering from", "You have", "Based on your symptoms, you likely have",
                "This indicates you are diagnosed with", "You present with findings consistent with",
            ],
            "report": [
                "Your lab results show", "I reviewed your MRI and it indicates", "Your CT scan demonstrates",
                "Your X-ray shows", "Interpreting your report, I find",
            ],
            "dosage": [
                "Take 400 mg every 8 hours", "You should take 1 tablet twice daily",
                "Increase your dosage to", "Reduce your dose to", "Apply this cream twice daily",
            ],
            "plan": [
                "I recommend you start", "We will start you on", "You should begin this treatment plan",
                "I will prescribe", "Given your case, you should follow",
            ],
            "whitelist": [
                "Common treatments for asthma include", "General information about diabetes is",
                "Typical side effects may include", "In general, treatment options are",
                "People should seek medical care if they experience",
            ],
        }

    def _load_seed_texts_from_json(self, path: str) -> Dict[str, List[str]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        defaults = self._default_seed_texts()
        for k in ALL_KEYS:
            if k in data and isinstance(data[k], list) and data[k]:
                defaults[k] = data[k]
        return defaults


def privacy_guard_output() -> OutputPrivacyGuard:
    return OutputPrivacyGuard()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Build and save output privacy prototypes (.pt)")
    parser.add_argument("--out", help="Path to save .pt (default: ./output_proto_artifacts/scope_prototypes.pt)")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--seeds", default=CUSTOM_SEEDS_JSON)
    parser.add_argument("--block-threshold", type=float, default=BLOCK_THRESHOLD)
    parser.add_argument("--whitelist-margin", type=float, default=WHITELIST_MARGIN)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output_proto_artifacts",
        "scope_prototypes.pt",
    )
    out_path = os.path.abspath(args.out or default_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and not args.force:
        print(f"[INFO] File already exists: {out_path}")
        print("Use --force to overwrite.\n")
        sys.exit(0)

    print("=" * 70)
    print("Building Output Privacy Prototypes (.pt)")
    print("=" * 70)
    print(f"Model      : {args.model}")
    print(f"Output path: {out_path}\n")

    guard = OutputPrivacyGuard(
        model_name=args.model,
        proto_path=None,
        block_threshold=args.block_threshold,
        whitelist_margin=args.whitelist_margin,
        seeds_json=args.seeds,
    )

    if not _HAS_TORCH:
        print("[ERROR] Torch not installed; cannot save .pt", file=sys.stderr)
        sys.exit(3)

    guard._save_prototypes_pt(out_path, guard.protos)

    vecs = guard.protos.vectors
    keys = list(vecs.keys())
    dim = next(iter(vecs.values())).shape[0] if vecs else -1
    print("[OK] prototypes saved")
    print(f"  model   : {guard.model_name}")
    print(f"  dim     : {dim}")
    print(f"  classes : {keys}")
    print(f"  path    : {out_path}")
    print("=" * 70)
