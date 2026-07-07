#!/usr/bin/env python
"""
Standalone prediction script for MultiModal-AFPpred.

Loads a trained checkpoint and predicts whether peptide sequences
are antifungal peptides (AFPs).

Usage:
    # Single sequence
    python predict.py --sequence "KWKLFKKILKVLNHV"

    # Batch prediction from a FASTA file
    python predict.py --fasta input.fasta --output predictions.csv

    # Interactive mode (enter sequences one by one)
    python predict.py --interactive

    # Specify custom checkpoint and ESM-2 paths
    python predict.py --sequence "KWKLFKKILKVLNHV" \
        --checkpoint checkpoints/best_model.pth \
        --esm-dir ESM2
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# Add src/ to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from pipeline import MultiModalAFPpred  # noqa: E402
from data_loader import _clean_seq, _length_filter, STANDARD_AA  # noqa: E402


# ──────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str) -> dict:
    """Load a saved checkpoint containing weights, standardizers, and config."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please run training first: python tests/run_final_training_demo.py"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")
    print(f"       Seed: {checkpoint.get('seed', 'N/A')}")
    if "test_metrics" in checkpoint:
        m = checkpoint["test_metrics"]
        auroc = m.get("auroc")
        mcc = m.get("mcc")
        if auroc is not None:
            print(f"       Test AUC-ROC: {auroc:.4f}")
        if mcc is not None:
            print(f"       Test MCC:     {mcc:.4f}")
    return checkpoint


def build_model_from_checkpoint(checkpoint: dict, esm_dir: str) -> MultiModalAFPpred:
    """Reconstruct the model architecture and load trained weights."""
    config = checkpoint.get("config", {})

    model = MultiModalAFPpred(
        model_dir=esm_dir,
        dim_model=config.get("dim_model", 512),
        physchem_dim=config.get("physchem_dim", 256),
        ss_dim=config.get("ss_dim", 128),
        n_heads=config.get("n_heads", 8),
        lstm_hidden=config.get("lstm_hidden", 128),
        classifier_hidden=tuple(config.get("classifier_hidden", [128, 64])),
        dropout=config.get("dropout", 0.1),
        classifier_return_logits=config.get("classifier_return_logits", True),
    )

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Restore feature standardizer parameters
    # These were fitted on the training set and are required for inference
    model.phys_standardizer.mean_ = checkpoint["phys_mean"]
    model.phys_standardizer.std_ = checkpoint["phys_std"]
    model.ss_standardizer.mean_ = checkpoint["ss_mean"]
    model.ss_standardizer.std_ = checkpoint["ss_std"]
    model._std_fitted = True

    return model


# ──────────────────────────────────────────────
# Sequence validation
# ──────────────────────────────────────────────

def validate_sequence(seq: str, min_len: int = 5, max_len: int = 50) -> str:
    """
    Clean and validate a peptide sequence.
    Returns the cleaned sequence or raises ValueError.
    """
    cleaned = _clean_seq(seq)
    if cleaned is None or len(cleaned) == 0:
        raise ValueError("Sequence is empty or contains invalid characters.")
    if len(cleaned) < min_len or len(cleaned) > max_len:
        raise ValueError(
            f"Sequence length {len(cleaned)} out of range "
            f"[{min_len}, {max_len}]."
        )
    return cleaned


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

@torch.no_grad()
def predict_sequences(
    model: MultiModalAFPpred,
    sequences: List[str],
    threshold: float,
    batch_size: int = 8,
) -> List[Tuple[str, float, str]]:
    """
    Run prediction on a list of peptide sequences.

    Returns list of (sequence, probability, label) tuples.
    label is "Antifungal" if prob >= threshold, else "Non-antifungal".
    """
    results = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i: i + batch_size]
        probs = model.predict_on_sequences(batch)
        probs_np = probs.squeeze(-1).cpu().numpy()

        for seq, prob in zip(batch, probs_np):
            label = "Antifungal" if float(prob) >= threshold else "Non-antifungal"
            results.append((seq, float(prob), label))

    return results


# ──────────────────────────────────────────────
# FASTA parsing
# ──────────────────────────────────────────────

def parse_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Parse a FASTA file into a list of (header, sequence) tuples.
    """
    entries = []
    header = None
    seq_lines = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_lines)))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

    if header is not None:
        entries.append((header, "".join(seq_lines)))

    return entries


# ──────────────────────────────────────────────
# Output formatting
# ──────────────────────────────────────────────

def print_prediction(seq: str, prob: float, label: str, threshold: float):
    """Print a single prediction result to console."""
    symbol = "+" if label == "Antifungal" else "-"
    print(f"  [{symbol}] {label:>15s}  |  prob={prob:.4f}  "
          f"(threshold={threshold:.2f})  |  {seq}")


def save_predictions_csv(
    results: List[Tuple[str, str, float, str]],
    output_path: str,
    threshold: float,
):
    """
    Save prediction results to a CSV file.
    results: list of (header, sequence, probability, label)
    """
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Header", "Sequence", "Probability", "Prediction", "Threshold"])
        for header, seq, prob, label in results:
            writer.writerow([header, seq, f"{prob:.6f}", label, f"{threshold:.4f}"])
    print(f"\n[INFO] Predictions saved to: {output_path}")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MultiModal-AFPpred: Predict antifungal peptides from sequences."
    )
    parser.add_argument(
        "--sequence", type=str, default=None,
        help="A single peptide sequence to predict.",
    )
    parser.add_argument(
        "--fasta", type=str, default=None,
        help="Path to a FASTA file for batch prediction.",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Interactive mode: enter sequences one by one.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file path (for FASTA batch mode).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="Path to the model checkpoint file (default: checkpoints/best_model.pth).",
    )
    parser.add_argument(
        "--esm-dir", type=str, default="ESM2",
        help="Path to the local ESM-2 model directory (default: ESM2).",
    )
    args = parser.parse_args()

    # At least one input mode must be specified
    if not args.sequence and not args.fasta and not args.interactive:
        parser.print_help()
        print("\nError: Please specify --sequence, --fasta, or --interactive.")
        sys.exit(1)

    # Load checkpoint and build model
    checkpoint = load_checkpoint(args.checkpoint)
    threshold = checkpoint.get("best_threshold", 0.5)
    model = build_model_from_checkpoint(checkpoint, args.esm_dir)

    print(f"[INFO] Decision threshold: {threshold:.4f}")
    print(f"[INFO] ESM-2 model dir: {args.esm_dir}")
    print()

    # ── Mode 1: Single sequence ──
    if args.sequence:
        try:
            seq = validate_sequence(args.sequence)
            print("Sequence:")
            results = predict_sequences(model, [seq], threshold)
            for s, p, l in results:
                print_prediction(s, p, l, threshold)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # ── Mode 2: FASTA batch ──
    if args.fasta:
        if not os.path.exists(args.fasta):
            print(f"Error: FASTA file not found: {args.fasta}")
            sys.exit(1)

        entries = parse_fasta(args.fasta)
        if not entries:
            print("Error: No sequences found in the FASTA file.")
            sys.exit(1)

        print(f"Loaded {len(entries)} sequences from {args.fasta}")
        print("-" * 70)

        valid_seqs = []
        valid_headers = []
        skipped = []
        for header, seq in entries:
            try:
                cleaned = validate_sequence(seq)
                valid_seqs.append(cleaned)
                valid_headers.append(header)
            except ValueError as e:
                skipped.append((header, str(e)))

        if skipped:
            print(f"[WARN] Skipped {len(skipped)} invalid sequences:")
            for h, reason in skipped:
                print(f"       {h}: {reason}")
            print()

        if valid_seqs:
            results = predict_sequences(model, valid_seqs, threshold)
            print("Predictions:")
            print("-" * 70)

            csv_results = []
            for i, (header, seq) in enumerate(zip(valid_headers, valid_seqs)):
                _, prob, label = results[i]
                print_prediction(seq, prob, label, threshold)
                csv_results.append((header, seq, prob, label))

            if args.output:
                save_predictions_csv(csv_results, args.output, threshold)

    # ── Mode 3: Interactive ──
    if args.interactive:
        print("Interactive mode — enter peptide sequences (Ctrl+C to exit).")
        print("-" * 70)
        while True:
            try:
                user_input = input("\nSequence> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                try:
                    seq = validate_sequence(user_input)
                    results = predict_sequences(model, [seq], threshold)
                    for s, p, l in results:
                        print_prediction(s, p, l, threshold)
                except ValueError as e:
                    print(f"  Error: {e}")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break


if __name__ == "__main__":
    main()
