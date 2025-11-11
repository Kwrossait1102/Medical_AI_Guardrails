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
    ):
        self.model_name = model_name
        self.block_threshold = block_threshold
        self.whitelist_margin = whitelist_margin
        self.model = SentenceTransformer(self.model_name)

        if proto_path and os.path.exists(proto_path):
            if proto_path.lower().endswith(".npz"):
                self.protos = self._load_prototypes_npz(proto_path)
            elif proto_path.lower().endswith(".pt"):
                if not _HAS_TORCH:
                    raise RuntimeError("Torch not installed; cannot load .pt")
                self.protos = self._load_prototypes_pt(proto_path)
            else:
                raise ValueError(f"Unsupported prototype file: {proto_path}")
        else:
            seeds = self._load_seed_texts_from_json(seeds_json) if seeds_json else self._default_seed_texts()
            self.protos = self._build_prototypes(seeds)
            if proto_path:
                os.makedirs(os.path.dirname(proto_path), exist_ok=True)
                if proto_path.lower().endswith(".npz"):
                    self._save_prototypes_npz(proto_path, self.protos)
                elif proto_path.lower().endswith(".pt"):
                    if not _HAS_TORCH:
                        raise RuntimeError("Torch not installed; cannot save .pt")
                    self._save_prototypes_pt(proto_path, self.protos)
                else:
                    self._save_prototypes_npz(proto_path + ".npz", self.protos)

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
        vecs: Dict[str, np.ndarray] = {}
        for cls, samples in seed_texts.items():
            if not samples:
                vecs[cls] = np.zeros((self._dim(),), dtype=np.float32)
                continue
            emb = self._encode(samples)
            proto = emb.mean(axis=0)
            proto = self._l2_normalize(proto)
            vecs[cls] = proto
        for k in ALL_KEYS:
            vecs.setdefault(k, np.zeros((self._dim(),), dtype=np.float32))
            seed_texts.setdefault(k, [])
        return ProtoSet(vectors=vecs, texts=seed_texts)

    def _save_prototypes_npz(self, path: str, protos: ProtoSet) -> None:
        np.savez_compressed(
            path,
            **{f"vec__{k}": v for k, v in protos.vectors.items()},
            **{f"txt__{k}": np.array(v) for k, v in protos.texts.items()},
            meta=np.array([self.model_name, self.block_threshold, self.whitelist_margin], dtype=object),
        )

    def _load_prototypes_npz(self, path: str) -> ProtoSet:
        data = np.load(path, allow_pickle=True)
        vectors: Dict[str, np.ndarray] = {}
        texts: Dict[str, List[str]] = {}
        for key in data.files:
            if key.startswith("vec__"):
                cls = key.split("__", 1)[1]
                vectors[cls] = data[key].astype(np.float32)
            elif key.startswith("txt__"):
                cls = key.split("__", 1)[1]
                texts[cls] = list(map(str, data[key].tolist()))
        for k in ALL_KEYS:
            vectors.setdefault(k, np.zeros((self._dim(),), dtype=np.float32))
            texts.setdefault(k, [])
        return ProtoSet(vectors=vectors, texts=texts)

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
        for k in ALL_KEYS:
            vectors.setdefault(k, np.zeros((self._dim(),), dtype=np.float32))
            texts.setdefault(k, [])
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
        embs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)

    def _dim(self) -> int:
        tmp = self._encode(["dim-probe"])
        return int(tmp.shape[1])

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-12
        return v / n

    def _default_seed_texts(self) -> Dict[str, List[str]]:
        return {
            "diagnosis": [
                "You are likely suffering from", "You have", "Based on your symptoms, you likely have",
                "This indicates you are diagnosed with", "You present with findings consistent with",
                "Your condition is consistent with", "It seems you might be affected by",
                "From your description, it sounds like", "These symptoms suggest you have",
                "I believe you are experiencing signs of", "This result indicates a possible case of",
                "Your symptoms align with those of", "This pattern is typical for patients with",
            ],
            "report": [
                "Your lab results show", "I reviewed your MRI and it indicates", "Your CT scan demonstrates",
                "Your X-ray shows", "Interpreting your report, I find", "Your blood test reveals",
                "The imaging results suggest", "According to your scan, there appears to be",
                "Your report indicates", "I analyzed your test results and found", "Your ECG shows abnormalities in",
                "Your medical report suggests", "Based on your laboratory results, it appears",
            ],
            "dosage": [
                "Take 400 mg every 8 hours", "You should take 1 tablet twice daily", "Your dose is 5 ml three times a day",
                "Increase your dosage to", "Reduce your dose to", "You can take up to 800 mg per day",
                "Apply this cream twice daily", "Take one pill before bed", "You need to take this medication every 6 hours",
                "Start with 10 mg once daily", "Your current dosage should be adjusted to", "You may need to lower the dose to",
                "Use two drops in each eye every 8 hours", "Take the medicine after meals", "The proper dose for you is",
            ],
            "plan": [
                "I recommend you start", "We will start you on", "You should begin this treatment plan",
                "I will prescribe", "Given your case, you should follow", "You should continue taking this medication",
                "I suggest you follow this regimen", "We can adjust your therapy to include", "You need to start this course of treatment",
                "Let’s initiate a new plan with", "You should combine medication and exercise", "You will start therapy next week",
                "I will add another drug to your regimen", "You should undergo surgery if symptoms persist",
                "Follow this plan until your next visit",
            ],
            "whitelist": [
                "Common treatments for asthma include", "General information about diabetes is",
                "Typical side effects may include", "In general, treatment options are",
                "People should seek medical care if they experience", "The usual adult dose for this drug class is described in textbooks",
                "Symptoms of flu often include fever and fatigue", "General prevention strategies involve regular exercise and balanced diet",
                "Doctors commonly treat hypertension using lifestyle modifications and medications",
                "Patients with mild symptoms are often advised to rest and hydrate",
                "This overview summarizes general medical approaches", "Public health guidelines recommend vaccination for prevention",
                "Research shows that early screening can reduce complications",
                "Clinical management typically depends on severity and comorbidities",
                "People should always consult professionals for diagnosis and treatment",
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
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output_proto_artifacts",
        "scope_prototypes.pt",
    )

    os.makedirs(os.path.dirname(default_out), exist_ok=True)

    guard = OutputPrivacyGuard(
        model_name=DEFAULT_EMBED_MODEL,
        proto_path=None,
        block_threshold=BLOCK_THRESHOLD,
        whitelist_margin=WHITELIST_MARGIN,
        seeds_json=CUSTOM_SEEDS_JSON,
    )

    if not _HAS_TORCH:
        raise RuntimeError("Torch not installed")

    guard._save_prototypes_pt(default_out, guard.protos)
    print(f"[OK] saved prototypes to {default_out}")
