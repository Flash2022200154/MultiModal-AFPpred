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
    keys = [
        "EISD860101","FASG760101","ZIMJ680103","TANS770101",
        "BURA740101","CHAM810101","GRAR740103","PONP930101","VASM830101"
    ]
    analyzer = AAIndexAnalyzer(keys)
    corr = analyzer.correlation_matrix()
    selected = analyzer.filter_by_threshold(threshold=threshold)
    return {"selected_keys": selected, "correlation": corr}