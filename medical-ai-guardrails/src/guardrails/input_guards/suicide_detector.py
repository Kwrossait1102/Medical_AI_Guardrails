import os
import json
import numpy as np
import pandas as pd
import torch
from typing import Optional
from sentence_transformers import SentenceTransformer, util

# Configuration
CSV_PATH = r"medical-ai-guardrails\src\guardrails\input_guards\Suicide_Detection.csv"
MODEL_NAME = "all-mpnet-base-v2"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training artifacts
ARTIFACT_DIR = "./suicide_proto_artifacts"
PROTO_PATH = os.path.join(ARTIFACT_DIR, "suicide_proto.pt")
META_PATH = os.path.join(ARTIFACT_DIR, "meta.json")

def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess the suicide detection dataset.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame with 'text' and 'label' columns
        
    Raises:
        ValueError: If required columns are not found
    """
    df = pd.read_csv(csv_path)
    
    text_col = "text" if "text" in df.columns else None
    label_col = next((c for c in ["class", "label", "target", "suicide"] if c in df.columns), None)
    
    if text_col is None or label_col is None:
        raise ValueError(f"Required columns not found. Current columns: {list(df.columns)}")
    
    df = df[[text_col, label_col]].dropna()
    
    # Normalize labels to 0/1
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].str.lower().map({"suicide": 1, "non-suicide": 0})
    df[label_col] = df[label_col].astype(int)
    
    df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)
    
    # Filter out extremely short texts
    df = df[df["text"].astype(str).str.len() >= 5]
    
    return df

def fit_one_prototype(
    df: pd.DataFrame,
    model_name: str = MODEL_NAME,
    max_pos: Optional[int] = 20000,
    max_neg: Optional[int] = 20000,
    neg_percentile: float = 95.0,
):
    """
    Train a single prototype for suicide detection using embeddings.
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        model_name: Name of the sentence transformer model
        max_pos: Maximum number of positive samples to use
        max_neg: Maximum number of negative samples to use
        neg_percentile: Percentile of non-suicide similarities to use as threshold
        
    Returns:
        Tuple of (suicide_prototype, threshold, model)
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    pos = df[df.label == 1]
    neg = df[df.label == 0]

    if max_pos and len(pos) > max_pos:
        pos = pos.sample(max_pos, random_state=42)
    if max_neg and len(neg) > max_neg:
        neg = neg.sample(max_neg, random_state=42)

    model = SentenceTransformer(model_name, device=DEVICE)

    def _embed(texts):
        return model.encode(texts, convert_to_tensor=True, batch_size=128, show_progress_bar=True)

    # Compute suicide prototype as normalized mean of positive embeddings
    pos_emb = _embed(pos.text.astype(str).tolist()).to(DEVICE)
    suicide_proto = util.normalize_embeddings(pos_emb.mean(dim=0, keepdim=True))

    # Calibrate threshold using non-suicide similarity distribution
    neg_emb = _embed(neg.text.astype(str).tolist()).to(DEVICE)
    neg_emb = util.normalize_embeddings(neg_emb)
    sims = util.cos_sim(neg_emb, suicide_proto).squeeze(1).detach().cpu().numpy()
    threshold = float(np.percentile(sims, neg_percentile))

    # Save artifacts
    torch.save({"suicide_proto": suicide_proto}, PROTO_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "device": DEVICE,
            "num_pos": int(len(pos)),
            "num_neg_for_threshold": int(len(neg)),
            "threshold_percentile": neg_percentile,
            "threshold": threshold
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] Prototype and threshold generated: threshold (p{neg_percentile}% of non-suicide) = {threshold:.4f}")
    return suicide_proto, threshold, model


class OneProtoSuicideDetector:
    """
    Suicide risk detector using single prototype matching.
    
    Attributes:
        suicide_proto: Normalized embedding prototype for suicide-related text
        threshold: Similarity threshold for risk detection
        model: Sentence transformer model for encoding
    """
    
    def __init__(self, suicide_proto: torch.Tensor, threshold: float, model: SentenceTransformer):
        """
        Initialize the detector.
        
        Args:
            suicide_proto: Pre-computed suicide prototype embedding
            threshold: Similarity threshold for triggering alerts
            model: Pre-loaded sentence transformer model
        """
        self.suicide_proto = suicide_proto.to(DEVICE)
        self.threshold = float(threshold)
        self.model = model

    @torch.no_grad()
    def similarity(self, text: str) -> float:
        """
        Compute cosine similarity between input text and suicide prototype.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Cosine similarity score
        """
        if not text or not text.strip():
            return -1e9
        emb = self.model.encode([text], convert_to_tensor=True, device=DEVICE)
        emb = util.normalize_embeddings(emb)
        return util.cos_sim(emb, self.suicide_proto).item()

    def check(self, text: str) -> Optional[str]:
        """
        Check if text indicates suicide risk.
        
        Args:
            text: Input text to check
            
        Returns:
            Crisis support message if risk detected, None otherwise
        """
        sim = self.similarity(text)
        if sim > self.threshold:
            return (
                "I'm concerned you might be at risk. "
                "If you are in immediate danger, please call your local emergency number. "
                "In Germany, dial 112. You can also reach TelefonSeelsorge at 116 123 (24/7). "
                "You matter, and help is available."
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
