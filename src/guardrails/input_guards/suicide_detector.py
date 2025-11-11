import os
import json
import numpy as np
import pandas as pd
import torch
from typing import Optional
from sentence_transformers import SentenceTransformer, util

CSV_PATH = r"medical-ai-guardrails\src\guardrails\input_guards\Suicide_Detection.csv"
MODEL_NAME = "all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARTIFACT_DIR = "./suicide_proto_artifacts"
PROTO_PATH = os.path.join(ARTIFACT_DIR, "suicide_proto.pt")
META_PATH = os.path.join(ARTIFACT_DIR, "meta.json")


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    text_col = "text" if "text" in df.columns else None
    label_col = next((c for c in ["class", "label", "target", "suicide"] if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Required columns not found: {list(df.columns)}")

    df = df[[text_col, label_col]].dropna()
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].str.lower().map({"suicide": 1, "non-suicide": 0})
    df[label_col] = df[label_col].astype(int)
    df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)
    return df[df["text"].astype(str).str.len() >= 5]


def fit_one_prototype(
    df: pd.DataFrame,
    model_name: str = MODEL_NAME,
    max_pos: Optional[int] = 20000,
    max_neg: Optional[int] = 20000,
    neg_percentile: float = 95.0,
):
    """Compute suicide prototype and threshold from dataset."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    pos, neg = df[df.label == 1], df[df.label == 0]

    if max_pos and len(pos) > max_pos:
        pos = pos.sample(max_pos, random_state=42)
    if max_neg and len(neg) > max_neg:
        neg = neg.sample(max_neg, random_state=42)

    model = SentenceTransformer(model_name, device=DEVICE)

    def _embed(texts):
        return model.encode(texts, convert_to_tensor=True, batch_size=128, show_progress_bar=True)

    pos_emb = _embed(pos.text.tolist()).to(DEVICE)
    suicide_proto = util.normalize_embeddings(pos_emb.mean(dim=0, keepdim=True))
    neg_emb = util.normalize_embeddings(_embed(neg.text.tolist()).to(DEVICE))
    sims = util.cos_sim(neg_emb, suicide_proto).squeeze(1).cpu().numpy()
    threshold = float(np.percentile(sims, neg_percentile))

    torch.save({"suicide_proto": suicide_proto}, PROTO_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "device": DEVICE,
            "num_pos": len(pos),
            "num_neg_for_threshold": len(neg),
            "threshold_percentile": neg_percentile,
            "threshold": threshold
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] Prototype generated: threshold (p{neg_percentile}%) = {threshold:.4f}")
    return suicide_proto, threshold, model


class OneProtoSuicideDetector:
    """Single-prototype suicide risk detector."""
    
    def __init__(self, suicide_proto: torch.Tensor, threshold: float, model: SentenceTransformer):
        self.suicide_proto = suicide_proto.to(DEVICE)
        self.threshold = float(threshold)
        self.model = model

    @torch.no_grad()
    def similarity(self, text: str) -> float:
        if not text or not text.strip():
            return -1e9
        emb = self.model.encode([text], convert_to_tensor=True, device=DEVICE)
        emb = util.normalize_embeddings(emb)
        return util.cos_sim(emb, self.suicide_proto).item()

    def check(self, text: str) -> Optional[str]:
        sim = self.similarity(text)
        if sim > self.threshold:
            return (
                "I'm concerned you might be at risk. "
                "If you are in immediate danger, please call your local emergency number. "
                "In Germany, dial 112. You can also reach TelefonSeelsorge at 116 123 (24/7)."
            )
        return None


if __name__ == "__main__":
    df = load_csv(CSV_PATH)
    proto, thresh, model = fit_one_prototype(df, neg_percentile=95.0)
    det = OneProtoSuicideDetector(proto, thresh, model)

    demo = "I don't want to be here anymore."
    sim = det.similarity(demo)
    msg = det.check(demo)
    
    print(f"text: {demo}\nsimilarity={sim:.4f}, threshold={thresh:.4f}")
    print("decision:", "RISK" if msg else "SAFE")
    if msg:
        print(msg)
