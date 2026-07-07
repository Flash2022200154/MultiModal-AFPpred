import numpy as np
import pandas as pd
from typing import List, Dict
from aaindex import aaindex1

ALLOWED_AA = list("ACDEFGHIKLMNPQRSTVWY")

class AAIndexAnalyzer:
    def __init__(self, keys: List[str]) -> None:
        self.keys = keys

    def build_matrix(self) -> pd.DataFrame:
        mat = {}
        for k in self.keys:
            vals = aaindex1[k].values
            mat[k] = [vals.get(aa, np.nan) for aa in ALLOWED_AA]
        return pd.DataFrame(mat, index=ALLOWED_AA)

    def correlation_matrix(self) -> pd.DataFrame:
        df = self.build_matrix()
        return df.corr(method="pearson")

    def filter_by_threshold(self, threshold: float = 0.8) -> List[str]:
        C = self.correlation_matrix().values
        P = len(self.keys)
        keep, removed = [], set()
        for i in range(P):
            if i in removed:
                continue
            keep.append(i)
            for j in range(i + 1, P):
                if j in removed:
                    continue
                if abs(C[i, j]) > threshold:
                    removed.add(j)
        return [self.keys[i] for i in keep]

def analyze_antimicrobial_aaindex(threshold: float = 0.8) -> Dict[str, object]:
    # Same 12 AMP-related AAindex keys as in enhanced_physchem_features.py
    keys = [
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
    analyzer = AAIndexAnalyzer(keys)
    corr = analyzer.correlation_matrix()
    selected = analyzer.filter_by_threshold(threshold=threshold)
    return {"selected_keys": selected, "correlation": corr}