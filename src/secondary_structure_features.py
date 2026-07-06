import numpy as np
from typing import List
import pywt
import warnings

ALLOWED_AA = list("ACDEFGHIKLMNPQRSTVWY")

# A lightweight heuristic mapping for 3-state SS propensity (H/E/C) as mock backend.
# These values are placeholders for development only.
# For production and paper reproducibility, use backend="netsurfp2" with NetSurfP-2.0.
MOCK_PROP = {
    # H,  E,  C  (sum ~ 1.0)
    "A": (0.45, 0.15, 0.40),
    "C": (0.10, 0.10, 0.80),
    "D": (0.10, 0.30, 0.60),
    "E": (0.40, 0.20, 0.40),
    "F": (0.20, 0.30, 0.50),
    "G": (0.05, 0.05, 0.90),
    "H": (0.20, 0.20, 0.60),
    "I": (0.35, 0.30, 0.35),
    "K": (0.35, 0.15, 0.50),
    "L": (0.45, 0.20, 0.35),
    "M": (0.40, 0.20, 0.40),
    "N": (0.10, 0.20, 0.70),
    "P": (0.05, 0.05, 0.90),
    "Q": (0.30, 0.20, 0.50),
    "R": (0.30, 0.15, 0.55),
    "S": (0.10, 0.15, 0.75),
    "T": (0.10, 0.20, 0.70),
    "V": (0.35, 0.30, 0.35),
    "W": (0.20, 0.30, 0.50),
    "Y": (0.15, 0.30, 0.55),
}

def _pad_to_length(seq: str, target_len: int) -> str:
    if len(seq) >= target_len:
        return seq
    if not seq:
        return "A" * target_len
    return seq + seq[-1] * (target_len - len(seq))

class SecondaryStructureFeatureExtractor:
    """
    Secondary structure feature extractor with pluggable backend.
    - backend: "mock" (default), "netsurfp2"
    - Output: fixed-length vector (target_dim) via wavelet projection of 3-state probabilities
    """
    def __init__(self, target_dim: int = 128, backend: str = "mock", wavelet: str = "db2"):
        self.target_dim = target_dim
        self.backend = backend
        self.wavelet = wavelet

    def _mock_predict(self, sequence: str) -> np.ndarray:
        # Build (L, 3) prob matrix for H/E/C
        seq = (sequence or "").upper()
        if not seq:
            seq = "A"
        probs = np.array([MOCK_PROP.get(aa, (0.33, 0.33, 0.34)) for aa in seq], dtype=float)  # (L,3)
        return probs

    def _backend_predict(self, sequence: str) -> np.ndarray:
        if self.backend == "mock":
            return self._mock_predict(sequence)
        if self.backend == "netsurfp2":
            warnings.warn(
                "NetSurfP-2.0 backend not yet integrated. "
                "Install NetSurfP-2.0 and implement the prediction call in "
                "secondary_structure_features.py:_backend_predict(). "
                "Falling back to mock backend."
            )
            return self._mock_predict(sequence)
        raise ValueError(f"Unknown secondary structure backend: {self.backend}")

    def extract(self, sequence: str, normalize: bool = False) -> np.ndarray:
        probs = self._backend_predict(sequence)  # (L, 3)
        # Flatten to 1D signal and wavelet-project to target_dim
        signal = probs.reshape(-1)  # (L*3,)
        coeffs = pywt.wavedec(signal, self.wavelet)
        all_c = np.concatenate(coeffs)
        if all_c.shape[0] < self.target_dim:
            out = np.pad(all_c, (0, self.target_dim - all_c.shape[0]))
        else:
            out = all_c[: self.target_dim]
        out = out.astype(np.float32)
        if normalize:
            m, s = out.mean(), out.std() + 1e-6
            out = (out - m) / s
        return out