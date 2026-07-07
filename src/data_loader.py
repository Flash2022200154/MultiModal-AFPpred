"""
Data loading and preprocessing utilities.

- Input: CSV with columns ["Peptide_Sequence", "label"]
- Preprocessing:
  1) Keep only sequences composed of 20 standard amino acids
  2) Length filter: 5 <= len(seq) <= 50
  3) Remove exact duplicates by sequence
- Output: (train_df, val_df)
"""

from typing import List, Tuple, Optional
import os
import pandas as pd
import random

STANDARD_AA = set(list("ACDEFGHIKLMNPQRSTVWY"))


def _clean_seq(seq: str) -> Optional[str]:
    if not isinstance(seq, str):
        return None
    s = seq.strip().upper()
    if not s:
        return None
    # Only keep sequences fully composed of standard amino acids
    if any(ch not in STANDARD_AA for ch in s):
        return None
    return s


def _length_filter(df: pd.DataFrame, seq_col: str, low: int, high: int) -> pd.DataFrame:
    return df[(df[seq_col].str.len() >= low) & (df[seq_col].str.len() <= high)].copy()


def _stratified_split(df: pd.DataFrame, label_col: str, val_size: float, random_state: int):
    rnd = random.Random(random_state)
    train_parts = []
    val_parts = []
    for label, part in df.groupby(label_col):
        idxs = list(part.index)
        rnd.shuffle(idxs)
        cut = int(len(idxs) * (1.0 - val_size))
        train_idx = idxs[:cut]
        val_idx = idxs[cut:]
        train_parts.append(df.loc[train_idx])
        val_parts.append(df.loc[val_idx])
    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, val_df


def load_data(
    file_path: str,
    seq_col: str = "Peptide_Sequence",
    label_col: str = "label",
    val_size: float = 0.2,
    random_state: int = 42,
    length_low: int = 5,
    length_high: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset from CSV and return (train_df, val_df) after preprocessing.

    Expected columns:
      - Peptide_Sequence: str
      - label: int in {0,1}

    Steps:
      1) Clean sequences (uppercase, standard amino acids only)
      2) Filter by length [length_low, length_high]
      3) Drop exact duplicates on sequence
      4) Stratified split by label into train/val

    Returns:
      (train_df, val_df): pandas DataFrames with columns [seq_col, label_col]
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Currently support CSV
    if not file_path.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported for now. Please provide a .csv file.")

    df = pd.read_csv(file_path)
    if seq_col not in df.columns:
        raise ValueError(f"Missing required column '{seq_col}' in CSV.")
    if label_col not in df.columns:
        raise ValueError(f"Missing required column '{label_col}' in CSV.")

    # 1) Clean and standardize sequences
    df = df[[seq_col, label_col]].copy()
    df[seq_col] = df[seq_col].apply(_clean_seq)
    df = df.dropna(subset=[seq_col]).reset_index(drop=True)

    # 2) Length filter
    df = _length_filter(df, seq_col, length_low, length_high)

    # 3) Exact duplicate removal
    df = df.drop_duplicates(subset=[seq_col]).reset_index(drop=True)

    # 4) Stratified split by label
    if df[label_col].nunique() == 1:
        # Fallback to random split when stratification is not possible
        idxs = list(df.index)
        random.Random(random_state).shuffle(idxs)
        cut = int(len(idxs) * (1.0 - val_size))
        train_df = df.iloc[idxs[:cut]].reset_index(drop=True)
        val_df = df.iloc[idxs[cut:]].reset_index(drop=True)
    else:
        train_df, val_df = _stratified_split(df, label_col, val_size, random_state)

    return train_df, val_df