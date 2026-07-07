import torch
from torch.utils.data import Dataset


class TripleFeatureDataset(Dataset):
    """
    Dataset for tri-modal features (ESM, PhysChem, SecondaryStructure).
    - esm_vecs: Tensor (N, esm_dim)
    - phys_vecs: Tensor (N, physchem_dim)
    - ss_vecs:   Tensor (N, ss_dim)
    - labels:    Tensor (N,)
    """
    def __init__(self, esm_vecs: torch.Tensor, phys_vecs: torch.Tensor, ss_vecs: torch.Tensor, labels: torch.Tensor):
        n = labels.shape[0]
        assert esm_vecs.shape[0] == phys_vecs.shape[0] == ss_vecs.shape[0] == n, "Size mismatch among inputs"
        self.esm = esm_vecs.float()
        self.phys = phys_vecs.float()
        self.ss = ss_vecs.float()
        self.labels = labels.float()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.esm[idx], self.phys[idx], self.ss[idx], self.labels[idx]
