import numpy as np
import pywt
from typing import List, Optional
from aaindex import aaindex1

# Antimicrobial-related AAindex keys (Table S1 in the paper)
DEFAULT_AMP_KEYS = [
    "KYTJ820101",  # Hydrophobicity (Kyte-Doolittle)
    "GRAR740102",  # Polarity (Grantham)
    "JANJ780101",  # Surface accessibility (Janin)
    "FAUJ880103",  # Van der Waals volume (Fauchere)
    "ZIMJ680104",  # Bulkiness (Zimmerman)
    "KLEP840101",  # Charge (Klein)
    "RICJ880107",  # Protonation tendency (Richardson)
    "BHAR880101",  # Backbone flexibility (Bhaskaran-Ponnuswamy)
    "CHOP780101",  # Conformational constraint (Chothia)
    "CHOP780203",  # Turn propensity (Chothia)
    "MIYS990101",  # Folding free energy contribution (Miyazawa-Jernigan)
    "ENGD860101",  # Solvation free energy (Eisenberg-McLachlan)
]

ALLOWED_AA = list("ACDEFGHIKLMNPQRSTVWY")

def _prop_vector_for_key(key: str) -> np.ndarray:
    vals = aaindex1[key].values
    return np.array([vals.get(aa, np.nan) for aa in ALLOWED_AA], dtype=float)

def _select_by_correlation(keys: List[str], threshold: float = 0.8) -> List[str]:
    if len(keys) <= 1:
        return keys
    # Build matrix: (20 AAs) x (num_props)
    M = np.stack([_prop_vector_for_key(k) for k in keys], axis=1)  # (20, P)
    # Impute NaN with column means
    col_means = np.nanmean(M, axis=0, keepdims=True)
    idx = np.where(np.isnan(M))
    M[idx] = np.take_along_axis(col_means, np.array([idx[1]]), axis=1)
    # Pearson corr across properties
    C = np.corrcoef(M, rowvar=False)  # (P, P)
    P = len(keys)
    keep = []
    removed = set()
    for i in range(P):
        if i in removed:
            continue
        keep.append(i)
        for j in range(i + 1, P):
            if j in removed:
                continue
            if abs(C[i, j]) > threshold:
                removed.add(j)
    return [keys[i] for i in keep]

def _pad_to_length(seq: str, target_len: int) -> str:
    if len(seq) >= target_len:
        return seq
    if not seq:
        return "A" * target_len
    return seq + seq[-1] * (target_len - len(seq))

class EnhancedPhysChemExtractor:
    """
    Enhanced AAindex-based physicochemical feature extractor:
    - Supports selecting antimicrobial-related properties and correlation filtering
    - Multi-scale sliding windows + wavelet projection to target_dim
    - Optional standardization can be done externally (e.g., FeatureStandardizer)
    """
    def __init__(
        self,
        target_dim: int = 256,
        window_sizes: List[int] = [3, 5, 7],
        wavelet: str = "db4",
        aaindex_keys: Optional[List[str]] = None,
        corr_threshold: float = 0.8,
    ) -> None:
        self.target_dim = target_dim
        self.window_sizes = window_sizes
        self.wavelet = wavelet
        base_keys = aaindex_keys if aaindex_keys is not None else DEFAULT_AMP_KEYS
        self.keys = _select_by_correlation(base_keys, threshold=corr_threshold)
        # Pre-fetch property dictionaries for speed
        self.prop_dicts = [aaindex1[k].values for k in self.keys]
        self.num_props = len(self.keys)

    def _window_signal(self, seq: str, w: int) -> np.ndarray:
        seq = (seq or "").upper()
        if any(ch not in ALLOWED_AA for ch in seq):
            # leave NaN for unknown AA then impute with zeros
            pass
        L = len(seq)
        if L < w:
            seq = _pad_to_length(seq, w)
            L = len(seq)
        feats = []
        for i in range(L - w + 1):
            window = seq[i : i + w]
            vec = []
            for aa in window:
                for d in self.prop_dicts:
                    vec.append(d.get(aa, np.nan))
            feats.append(vec if vec else [0.0] * (self.num_props * w))
        if not feats:
            feats = [[0.0] * (self.num_props * w)]
        signal = np.array(feats, dtype=float).flatten()
        # Impute NaNs with zero
        if np.isnan(signal).any():
            col_mean = np.nanmean(signal)
            if np.isnan(col_mean):
                col_mean = 0.0
            signal = np.nan_to_num(signal, nan=col_mean)
        return signal

    def extract(self, sequence: str, normalize: bool = False) -> np.ndarray:
        # Multi-scale window -> wavelet -> concat to target_dim
        n_scales = max(1, len(self.window_sizes))
        per_dim = self.target_dim // n_scales
        outs = []
        for w in self.window_sizes:
            sig = self._window_signal(sequence, w)
            coeffs = pywt.wavedec(sig, self.wavelet)
            all_c = np.concatenate(coeffs)
            if all_c.shape[0] < per_dim:
                all_c = np.pad(all_c, (0, per_dim - all_c.shape[0]))
            else:
                all_c = all_c[:per_dim]
            outs.append(all_c.astype(np.float32))
        out = np.concatenate(outs, axis=0)
        if out.shape[0] < self.target_dim:
            out = np.pad(out, (0, self.target_dim - out.shape[0]))
        elif out.shape[0] > self.target_dim:
            out = out[: self.target_dim]
        if normalize:
            # simple z-norm on-the-fly (use external standardizer in training for consistency)
            m, s = out.mean(), out.std() + 1e-6
            out = (out - m) / s
        return out.astype(np.float32)