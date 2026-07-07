"""
Repeated training + evaluation with confidence intervals:
- 5 independent runs with different random seeds
- Train with early stopping using Val Accuracy
- Find best threshold on validation (max F1)
- Evaluate on test with the best validation threshold
- Report mean, SD, and 95% confidence intervals
"""
import sys
import os
import random
import math
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
from typing import Optional, Dict, List
# Put src to the front
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import MultiModalAFPpred  # noqa: E402
from datasets import TripleFeatureDataset  # noqa: E402
from data_loader import _clean_seq, _length_filter, STANDARD_AA  # noqa: E402


# ──────────────────────────────────────────────
# 0) Reproducibility utilities
# ──────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# 1) Metrics
# ──────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(np.float32)
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)
    spec = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float("nan")
    return {
        "sensitivity": sens,
        "specificity": spec,
        "accuracy": acc,
        "mcc": mcc,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
    }


# ──────────────────────────────────────────────
# 2) Training helpers
# ──────────────────────────────────────────────

@torch.no_grad()
def predict_proba(model: "MultiModalAFPpred", loader: DataLoader) -> np.ndarray:
    model.eval()
    probs_all = []
    for esm_vecs, phys_vecs, ss_vecs, _labels in loader:
        logits = model(esm_vecs, phys_vecs, ss_vecs).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)


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


def train_one_epoch(model, loader, opt, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for esm_vecs, phys_vecs, ss_vecs, labels in loader:
        logits = model(esm_vecs, phys_vecs, ss_vecs).squeeze(-1)
        loss = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return total_loss / max(1, n)


# ──────────────────────────────────────────────
# 3) Data loading
# ──────────────────────────────────────────────

def is_standard_seq(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip().upper()
    if not s:
        return False
    return all(ch in STANDARD_AA for ch in s)


def detect_sequence_column(df: pd.DataFrame) -> Optional[str]:
    best_col = None
    best_ratio = 0.0
    for col in df.columns:
        series = df[col].astype(str)
        total = len(series)
        if total == 0:
            continue
        ok = series.apply(is_standard_seq).sum()
        ratio = ok / max(1, total)
        if ratio > best_ratio and ok >= 10:
            best_ratio = ratio
            best_col = col
    return best_col


def load_full_dataframe(max_pos: int = None, max_neg: int = None) -> "pd.DataFrame":
    """
    Build a labeled dataset from data_positive.csv and data_negative.csv.
    Data loading uses fixed seed (42) for consistent dataset across runs.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"

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

    df_pos = pd.read_csv(pos_path)
    pos_col = detect_sequence_column(df_pos)
    if pos_col is None:
        raise ValueError(f"No standard amino acid sequence column found in {pos_path}")
    pos_seqs = [str(s).strip().upper() for s in df_pos[pos_col].tolist() if is_standard_seq(s)]

    df_neg = pd.read_csv(neg_path)
    neg_col = detect_sequence_column(df_neg)
    if neg_col is None:
        raise ValueError(f"No standard amino acid sequence column found in {neg_path}")
    neg_seqs = [str(s).strip().upper() for s in df_neg[neg_col].tolist() if is_standard_seq(s)]

    pos_seqs = list(dict.fromkeys(pos_seqs))
    neg_seqs = list(dict.fromkeys(neg_seqs))

    pos_set = set(pos_seqs)
    neg_set = set(neg_seqs)
    conflict = pos_set & neg_set
    if conflict:
        pos_seqs = [s for s in pos_seqs if s not in conflict]
        neg_seqs = [s for s in neg_seqs if s not in conflict]

    if max_pos is not None:
        pos_seqs = pos_seqs[:max_pos]
    if max_neg is not None:
        neg_seqs = neg_seqs[:max_neg]

    df_all = pd.DataFrame({
        "Peptide_Sequence": pos_seqs + neg_seqs,
        "label": [1] * len(pos_seqs) + [0] * len(neg_seqs)
    })

    df_all["Peptide_Sequence"] = df_all["Peptide_Sequence"].apply(_clean_seq)
    df_all = df_all.dropna(subset=["Peptide_Sequence"]).reset_index(drop=True)
    df_all = _length_filter(df_all, "Peptide_Sequence", low=5, high=50)
    df_all = df_all.drop_duplicates(subset=["Peptide_Sequence"]).reset_index(drop=True)

    num_pos = (df_all["label"] == 1).sum()
    num_neg = (df_all["label"] == 0).sum()
    if num_pos == 0 or num_neg == 0:
        warnings.warn("Single class data, cannot perform stratified split.")
        return df_all
    minority = min(num_pos, num_neg)
    df_pos = df_all[df_all["label"] == 1].sample(n=minority, random_state=42)
    df_neg = df_all[df_all["label"] == 0].sample(n=minority, random_state=42)
    df_balanced = pd.concat([df_pos, df_neg], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df_balanced


# ──────────────────────────────────────────────
# 4) Single experiment run
# ──────────────────────────────────────────────

def run_single_experiment(
    seed: int,
    esm_dir: str,
    max_epochs: int = 200,
    patience: int = 4,
    batch_size: int = 128,
    lr: float = 1e-4,
) -> Dict[str, float]:
    """
    Run one complete training + evaluation cycle.

    Returns dict with keys: seed, accuracy, sensitivity, specificity,
    mcc, f1, auroc, auprc, best_thr
    """
    set_seed(seed)

    # Load and split data (varying seed controls train/val/test split)
    df = load_full_dataframe(max_pos=None, max_neg=None)

    trainval_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.2, random_state=seed, stratify=trainval_df["label"]
    )

    print(f"\n{'='*60}")
    print(f"  Run seed = {seed}  |  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"{'='*60}")

    # Build model (fresh initialization; seed controls initial weights)
    model = MultiModalAFPpred(model_dir=esm_dir, classifier_return_logits=True)

    tr_seqs = train_df["Peptide_Sequence"].tolist()
    va_seqs = val_df["Peptide_Sequence"].tolist()
    te_seqs = test_df["Peptide_Sequence"].tolist()
    tr_labels = torch.tensor(train_df["label"].values, dtype=torch.float32)
    va_labels = torch.tensor(val_df["label"].values, dtype=torch.float32)
    te_labels = torch.tensor(test_df["label"].values, dtype=torch.float32)

    model.fit_standardizers(tr_seqs)

    tr_esm, tr_phys, tr_ss = model.encode_sequences_to_features(tr_seqs)
    va_esm, va_phys, va_ss = model.encode_sequences_to_features(va_seqs)
    te_esm, te_phys, te_ss = model.encode_sequences_to_features(te_seqs)

    train_ds = TripleFeatureDataset(tr_esm, tr_phys, tr_ss, tr_labels)
    val_ds = TripleFeatureDataset(va_esm, va_phys, va_ss, va_labels)
    test_ds = TripleFeatureDataset(te_esm, te_phys, te_ss, te_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=4, min_lr=1e-5
    )

    best_acc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        with torch.no_grad():
            val_probs = predict_proba(model, val_loader)
            val_preds = (val_probs >= 0.5).astype(int)
            val_acc = accuracy_score(val_df["label"].values.astype(int), val_preds)
            val_loss = evaluate_loss(model, val_loader, criterion)

        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]["lr"]

        if val_acc > best_acc + 1e-6:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Best threshold on validation (max F1)
    with torch.no_grad():
        val_probs = predict_proba(model, val_loader)
    y_val = val_df["label"].values.astype(int)
    thrs = np.arange(0.0, 1.001, 0.01)
    f1s = [f1_score(y_val, (val_probs >= t).astype(int)) for t in thrs]
    best_thr = float(thrs[int(np.argmax(f1s))])

    # Test evaluation
    with torch.no_grad():
        test_probs = predict_proba(model, test_loader)
    y_test = test_df["label"].values.astype(int)
    mets = compute_metrics(y_test, test_probs, thr=best_thr)

    print(f"  [seed={seed}]  AUC-ROC={mets['auroc']:.4f}  MCC={mets['mcc']:.4f}  F1={mets['f1']:.4f}")

    return {"seed": seed, **mets, "best_thr": best_thr}


# ──────────────────────────────────────────────
# 5) Main: repeated runs + summary
# ──────────────────────────────────────────────

def main():
    max_epochs = 200
    patience = 4
    batch_size = 128
    lr = 1e-4

    esm_dir = str(Path(__file__).resolve().parents[2] / "ESM2")
    seeds = [42, 123, 456, 789, 1024]

    all_results: List[Dict[str, float]] = []

    print("=" * 60)
    print("  REPEATED RUNS — Confidence Interval Evaluation")
    print(f"  Seeds: {seeds}")
    print("=" * 60)

    for seed in seeds:
        result = run_single_experiment(
            seed=seed, esm_dir=esm_dir,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, lr=lr,
        )
        all_results.append(result)

    # ── Aggregate statistics ──
    metric_order = [
        ("auroc", "AUC-ROC"),
        ("auprc", "AUC-PR"),
        ("sensitivity", "Sens"),
        ("accuracy", "Acc"),
        ("f1", "F1"),
        ("specificity", "Spec"),
        ("mcc", "MCC"),
    ]

    n_runs = len(seeds)
    rows = []

    for key, label in metric_order:
        vals = np.array([r[key] for r in all_results])
        mean = vals.mean()
        std = vals.std(ddof=1)  # sample std
        ci_margin = 1.96 * std / math.sqrt(n_runs)  # 95% CI, normal approximation
        row = {"Metric": label}
        for i in range(n_runs):
            row[f"Run {i+1}"] = f"{vals[i]:.4f}"
        row["Mean ± SD"] = f"{mean:.4f} ± {std:.4f}"
        row["95% CI"] = f"[{mean - ci_margin:.4f}, {mean + ci_margin:.4f}]"
        rows.append(row)

    df_summary = pd.DataFrame(rows)

    # ── Console output ──
    print("\n")
    print("=" * 100)
    print("  CONFIDENCE INTERVALS FROM REPEATED RUNS")
    print("=" * 100)
    print(df_summary.to_string(index=False))
    print("=" * 100)

    # ── Save to CSV ──
    results_dir = Path(__file__).resolve().parents[2] / "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = results_dir / "repeated_runs_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n[INFO] Summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
