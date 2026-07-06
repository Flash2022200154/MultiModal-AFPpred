"""
Physicochemical features with AAIndex1 + multi-scale sliding window + wavelet (db4).

- Twelve antimicrobial-related AAIndex1 properties (Table S1):
  KYTJ820101 (hydrophobicity), GRAR740102 (polarity), JANJ780101 (surface accessibility),
  FAUJ880103 (van der Waals volume), ZIMJ680104 (bulkiness), KLEP840101 (charge),
  RICJ880107 (protonation tendency), BHAR880101 (backbone flexibility),
  CHOP780101 (conformational constraint), CHOP780203 (turn propensity),
  MIYS990101 (folding free energy), ENGD860101 (solvation free energy)
- Multi-scale windows (default [3,5,7]) to capture local context
- Wavelet decomposition with 'db4' and concat to target_dim=256

Note:
- Non-standard amino acids are mapped to np.nan via dict.get(aa, np.nan).
- Downstream pipeline should handle imputation/standardization (as in notebook).
"""
from typing import List
import numpy as np
import pywt
from aaindex import aaindex1

# AAIndex1 keys from Table S1 in the paper
PHYSCHEM_KEYS = [
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
N_PHYSCHEM = len(PHYSCHEM_KEYS)

# Load property dictionaries
PHYSCHEM_DICTS = [aaindex1[key].values for key in PHYSCHEM_KEYS]

ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")  # 20 standard AAs


def _pad_to_length(seq: str, target_len: int) -> str:
    """Pad sequence to target_len by repeating the last residue (as in the notebook)."""
    if len(seq) >= target_len:
        return seq
    if not seq:
        return "A" * target_len  # fallback if empty
    return seq + seq[-1] * (target_len - len(seq))


def _window_features(seq: str, window_size: int) -> np.ndarray:
    """
    Build sliding-window local features:
    - For each position, take a window of size w
    - Map each residue to N_PHYSCHEM properties from Table S1 indices and concat
    - Return flattened 1D signal
    """
    L = len(seq)
    if L < window_size:
        seq = _pad_to_length(seq, window_size)
        L = len(seq)

    local_feats: List[List[float]] = []
    n_props = N_PHYSCHEM
    for i in range(L - window_size + 1):
        window = seq[i : i + window_size]
        vec = []
        for aa in window:
            for prop_dict in PHYSCHEM_DICTS:
                vec.append(prop_dict.get(aa, np.nan))
        local_feats.append(vec if vec else [0.0] * (n_props * window_size))

    if not local_feats:
        local_feats = [[0.0] * (n_props * window_size)]

    signal = np.array(local_feats, dtype=float).flatten()
    return signal


def get_physchem_wavelet_multi(
    sequence: str,
    window_sizes: List[int] = [3, 5, 7],
    wavelet: str = "db4",
    target_dim: int = 256,
) -> np.ndarray:
    """
    Compute 256-dim physicochemical feature vector using multi-scale windows and wavelet.

    Args:
        sequence: peptide sequence (uppercase letters of 20 AAs recommended)
        window_sizes: scales to use (default [3,5,7])
        wavelet: wavelet family name (default 'db4')
        target_dim: final feature dimension (default 256)

    Returns:
        np.ndarray of shape (target_dim,), dtype float32
        (May contain NaNs if non-standard AAs are present; downstream should impute)
    """
    seq = (sequence or "").upper()
    # Pad to the longest window size for stability
    max_w = max(window_sizes) if window_sizes else 5
    seq = _pad_to_length(seq, max_w)

    scale_feats: List[np.ndarray] = []
    n_scales = max(1, len(window_sizes))
    per_dim = target_dim // n_scales

    for w in window_sizes:
        signal = _window_features(seq, w)

        # Wavelet decomposition and concat all levels' coefficients
        coeffs = pywt.wavedec(signal, wavelet)
        all_c = np.concatenate(coeffs)

        # Truncate or zero-pad to per_dim
        if all_c.shape[0] < per_dim:
            pad_len = per_dim - all_c.shape[0]
            all_c = np.concatenate([all_c, np.zeros(pad_len, dtype=float)])
        else:
            all_c = all_c[:per_dim]

        scale_feats.append(all_c.astype(np.float32))

    out = np.concatenate(scale_feats)
    # If target_dim is not divisible by #scales, pad to target_dim
    if out.shape[0] < target_dim:
        out = np.concatenate([out, np.zeros(target_dim - out.shape[0], dtype=np.float32)])
    elif out.shape[0] > target_dim:
        out = out[:target_dim]
    return out.astype(np.float32)