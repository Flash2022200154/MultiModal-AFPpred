"""
Final training + evaluation demo:
- Train with early stopping using Val Accuracy
- After training, find best threshold on validation (max F1)
- Evaluate on test with the best validation threshold
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, matthews_corrcoef,
    roc_auc_score, average_precision_score
)
import warnings
from typing import Optional
# Put src to the front
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import MultiModalAFPpred  # noqa: E402
from datasets import FeatureDataset  # noqa: E402
from data_loader import _clean_seq, _length_filter, STANDARD_AA  # noqa: E402
from datasets import TripleFeatureDataset  # noqa: E402


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(np.float32)
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)  # sensitivity
    spec = recall_score(y_true, y_pred, pos_label=0)  # specificity
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    # AUROC/AUPRC based on probabilities
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float("nan")
    return {
        "acc": acc, "sens": sens, "spec": spec, "f1": f1, "mcc": mcc, "auroc": auroc, "auprc": auprc
    }


@torch.no_grad()
def predict_proba(model: "MultiModalAFPpred", loader: DataLoader) -> np.ndarray:
    model.eval()
    probs_all = []
    for esm_vecs, phys_vecs, ss_vecs, _labels in loader:
        logits = model(esm_vecs, phys_vecs, ss_vecs).squeeze(-1)  # (B,)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)


def train_one_epoch(model, loader, opt, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for esm_vecs, phys_vecs, ss_vecs, labels in loader:
        logits = model(esm_vecs, phys_vecs, ss_vecs).squeeze(-1)  # (B,)
        loss = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return total_loss / max(1, n)


def is_standard_seq(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip().upper()
    if not s:
        return False
    return all(ch in STANDARD_AA for ch in s)


def detect_sequence_column(df: pd.DataFrame) -> Optional[str]:
    # Select the column with the highest standard AA ratio as the sequence column
    best_col = None
    best_ratio = 0.0
    for col in df.columns:
        series = df[col].astype(str)
        total = len(series)
        if total == 0:
            continue
        ok = series.apply(is_standard_seq).sum()
        ratio = ok / max(1, total)
        if ratio > best_ratio and ok >= 10:  # at least 10 to avoid noise columns
            best_ratio = ratio
            best_col = col
    return best_col



def load_full_dataframe(max_pos: int = None, max_neg: int = None) -> "pd.DataFrame":
    """
    Build a labeled dataset from data_positive.csv and data_negative.csv.

    Looks for the CSV files under the repo root/data/ directory.
    Auto-detects sequence columns, filters non-standard AA and length 5-50,
    removes exact duplicates.

    Returns DataFrame[Peptide_Sequence, label]
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"

    # Locate positive/negative CSV files
    pos_candidates = [data_dir / "data_positive.csv", repo_root / "data_positive.csv"]
    neg_candidates = [data_dir / "data_negative.csv", repo_root / "data_negative.csv"]

    pos_path = None
    for p in pos_candidates:
        if p.exists():
            pos_path = p
            break
    if pos_path is None:
        raise FileNotFoundError(
            "data_positive.csv not found. Expected locations:\n"
            + "\n".join(f"  {p}" for p in pos_candidates)
        )

    neg_path = None
    for p in neg_candidates:
        if p.exists():
            neg_path = p
            break
    if neg_path is None:
        raise FileNotFoundError(
            "data_negative.csv not found. Expected locations:\n"
            + "\n".join(f"  {p}" for p in neg_candidates)
        )

    # Read positive CSV
    df_pos = pd.read_csv(pos_path)
    pos_col = detect_sequence_column(df_pos)
    if pos_col is None:
        raise ValueError(f"No standard amino acid sequence column found in {pos_path}")
    pos_seqs = [str(s).strip().upper() for s in df_pos[pos_col].tolist() if is_standard_seq(s)]
    print(f"[INFO] data_positive.csv: read {len(pos_seqs)} sequences (col={pos_col})")

    # Read negative CSV
    df_neg = pd.read_csv(neg_path)
    neg_col = detect_sequence_column(df_neg)
    if neg_col is None:
        raise ValueError(f"No standard amino acid sequence column found in {neg_path}")
    neg_seqs = [str(s).strip().upper() for s in df_neg[neg_col].tolist() if is_standard_seq(s)]
    print(f"[INFO] data_negative.csv: read {len(neg_seqs)} sequences (col={neg_col})")

    # Deduplicate within each class
    pos_seqs = list(dict.fromkeys(pos_seqs))
    neg_seqs = list(dict.fromkeys(neg_seqs))

    # Remove sequences that appear in both positive and negative sets (conflicting labels)
    pos_set = set(pos_seqs)
    neg_set = set(neg_seqs)
    conflict = pos_set & neg_set
    if conflict:
        print(f"[WARN] {len(conflict)} sequences appear in both positive and negative sets; removing from both")
        pos_seqs = [s for s in pos_seqs if s not in conflict]
        neg_seqs = [s for s in neg_seqs if s not in conflict]

    # Optional: trim counts
    if max_pos is not None:
        pos_seqs = pos_seqs[:max_pos]
    if max_neg is not None:
        neg_seqs = neg_seqs[:max_neg]

    print(f"[INFO] After aggregation: positive={len(pos_seqs)}, negative={len(neg_seqs)}")

    # Build DataFrame
    df_all = pd.DataFrame({
        "Peptide_Sequence": pos_seqs + neg_seqs,
        "label": [1] * len(pos_seqs) + [0] * len(neg_seqs)
    })

    # Standard cleaning pipeline (consistent with data_loader)
    df_all["Peptide_Sequence"] = df_all["Peptide_Sequence"].apply(_clean_seq)
    df_all = df_all.dropna(subset=["Peptide_Sequence"]).reset_index(drop=True)
    df_all = _length_filter(df_all, "Peptide_Sequence", low=5, high=50)
    df_all = df_all.drop_duplicates(subset=["Peptide_Sequence"]).reset_index(drop=True)

    # Balancing: downsample majority class
    num_pos = (df_all["label"] == 1).sum()
    num_neg = (df_all["label"] == 0).sum()
    if num_pos == 0 or num_neg == 0:
        warnings.warn("Single class data, cannot perform stratified split.")
        return df_all
    minority = min(num_pos, num_neg)
    df_pos = df_all[df_all["label"] == 1].sample(n=minority, random_state=42)
    df_neg = df_all[df_all["label"] == 0].sample(n=minority, random_state=42)
    df_balanced = pd.concat([df_pos, df_neg], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"[INFO] After balancing, total samples: {len(df_balanced)} (pos={minority}, neg={minority})")
    return df_balanced


def main():
    # 0) Hyperparameters
    max_epochs = 200
    patience = 4
    batch_size = 128
    lr = 1e-4

    # 1) Read data from data_positive.csv and data_negative.csv
    df = load_full_dataframe(
        max_pos=None,  # set to an integer (e.g., 3000) if runtime is a concern
        max_neg=None   # set to an integer (e.g., 3000) if runtime is a concern
    )

    # Three-way split (train/val/test, stratified: 64/16/20)
    from sklearn.model_selection import train_test_split
    trainval_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.2, random_state=42, stratify=trainval_df["label"]
    )

    print("Starting final model training...")
    print(f"[INFO] train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # 2) Build model (classifier outputs logits for BCEWithLogitsLoss)
    esm_dir = str(Path(__file__).resolve().parents[2] / "ESM2")
    model = MultiModalAFPpred(model_dir=esm_dir, classifier_return_logits=True)

    # 3) Fit standardizers (on training set)
    tr_seqs = train_df["Peptide_Sequence"].tolist()
    va_seqs = val_df["Peptide_Sequence"].tolist()
    te_seqs = test_df["Peptide_Sequence"].tolist()
    tr_labels = torch.tensor(train_df["label"].values, dtype=torch.float32)
    va_labels = torch.tensor(val_df["label"].values, dtype=torch.float32)
    te_labels = torch.tensor(test_df["label"].values, dtype=torch.float32)

    model.fit_standardizers(tr_seqs)

    # 4) Precompute tri-modal features
    tr_esm, tr_phys, tr_ss = model.encode_sequences_to_features(tr_seqs)
    va_esm, va_phys, va_ss = model.encode_sequences_to_features(va_seqs)
    te_esm, te_phys, te_ss = model.encode_sequences_to_features(te_seqs)

    # 5) DataLoader (tri-modal)
    train_ds = TripleFeatureDataset(tr_esm, tr_phys, tr_ss, tr_labels)
    val_ds = TripleFeatureDataset(va_esm, va_phys, va_ss, va_labels)
    test_ds = TripleFeatureDataset(te_esm, te_phys, te_ss, te_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 5) Training components
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=4, min_lr=1e-5
    )
    # 6) Training + early stopping (based on Val Accuracy), with val loss driving LR scheduling
    best_acc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        # Validation: Accuracy and Val Loss
        with torch.no_grad():
            val_probs = predict_proba(model, val_loader)
            val_preds = (val_probs >= 0.5).astype(int)
            val_acc = accuracy_score(val_df["label"].values.astype(int), val_preds)
            val_loss = evaluate_loss(model, val_loader, criterion)

        # Scheduler adjusts learning rate based on validation loss
        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{max_epochs} �?Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {curr_lr:.2e}")

        if val_acc > best_acc + 1e-6:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early Stop: Validation Accuracy did not improve, stopping training.")
                break

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # 7) Scan thresholds on validation set (maximize F1)
    with torch.no_grad():
        val_probs = predict_proba(model, val_loader)
    y_val = val_df["label"].values.astype(int)
    thrs = np.arange(0.0, 1.001, 0.01)
    f1s = [f1_score(y_val, (val_probs >= t).astype(int)) for t in thrs]
    best_idx = int(np.argmax(f1s))
    best_thr = float(thrs[best_idx])
    best_val_f1 = float(f1s[best_idx])

    print(f"\nValidation set best threshold: {best_thr:.2f}, corresponding F1: {best_val_f1:.4f}")

    # 8) Evaluate on test set
    with torch.no_grad():
        test_probs = predict_proba(model, test_loader)
    y_test = test_df["label"].values.astype(int)
    mets = compute_metrics(y_test, test_probs, thr=best_thr)

    print(f"\n=== Test Set Evaluation (threshold = {best_thr:.2f}) ===")
    print(f"Accuracy:           {mets['acc']:.4f}")
    print(f"Sensitivity (Sens): {mets['sens']:.4f}")
    print(f"Specificity (Spec): {mets['spec']:.4f}")
    print(f"F1 Score:           {mets['f1']:.4f}")
    print(f"MCC:                {mets['mcc']:.4f}")
    print(f"AUROC:              {mets['auroc']:.4f}")
    print(f"AUPRC:              {mets['auprc']:.4f}")


@torch.no_grad()
def evaluate_loss(model: "MultiModalAFPpred", loader: DataLoader, criterion) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for esm_vecs, phys_vecs, ss_vecs, labels in loader:
        logits = model(esm_vecs, phys_vecs, ss_vecs).squeeze(-1)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return total_loss / max(1, n)

if __name__ == "__main__":
    main()