"""
End-to-end tri-modal pipeline:
ESM-2 (640-dim) + Enhanced PhysChem (256-dim, AAIndex) + Secondary Structure (128-dim, NetSurfP-2.0)
-> HierarchicalTriModalFusion (512-dim) -> ResidualBiLSTM -> Classifier
"""
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn

from feature_extractor import ESM2Embedder
from sequence_model import ResidualBiLSTMBlock
from classifier import ClassifierHead


class MultiModalAFPpred(nn.Module):
    """
    ESM-2 + Enhanced PhysChem (AAindex with correlation filtering) + Secondary Structure (NetSurfP-2.0/mock)
    -> HierarchicalTriModalFusion -> ResidualBiLSTM -> Classifier
    """
    def __init__(
        self,
        model_dir: str = "./ESM2",
        dim_model: int = 512,
        physchem_dim: int = 256,
        ss_dim: int = 128,
        n_heads: int = 8,
        lstm_hidden: int = 128,
        classifier_hidden=(128, 64),
        dropout: float = 0.1,
        embedder: Optional[ESM2Embedder] = None,
        classifier_return_logits: bool = False,
        aaindex_keys: Optional[List[str]] = None,
        corr_threshold: float = 0.8,
        ss_backend: str = "netsurfp2",
    ) -> None:
        super().__init__()
        self.embedder = embedder or ESM2Embedder(model_dir=model_dir, local_files_only=True)
        assert self.embedder.hidden_size is not None, "ESM2 hidden_size not found."
        self.esm_dim = int(self.embedder.hidden_size)
        self.physchem_dim = physchem_dim
        self.ss_dim = ss_dim
        self.dim_model = dim_model

        from enhanced_physchem_features import EnhancedPhysChemExtractor
        from secondary_structure_features import SecondaryStructureFeatureExtractor
        from standardize import FeatureStandardizer
        self.phys_extractor = EnhancedPhysChemExtractor(
            target_dim=physchem_dim, aaindex_keys=aaindex_keys, corr_threshold=corr_threshold
        )
        self.ss_extractor = SecondaryStructureFeatureExtractor(
            target_dim=ss_dim, backend=ss_backend, wavelet="db2"
        )
        self.phys_standardizer = FeatureStandardizer()
        self.ss_standardizer = FeatureStandardizer()
        self._std_fitted = False

        from fusion import HierarchicalTriModalFusion
        self.fusion = HierarchicalTriModalFusion(
            dim_esm=self.esm_dim, dim_phys=physchem_dim, dim_ss=ss_dim,
            dim_model=dim_model, n_heads=n_heads, attn_dropout=dropout, proj_dropout=dropout, dropout=dropout
        )
        self.seq_block = ResidualBiLSTMBlock(
            fusion_dim=dim_model, conv_filters=64, lstm_hidden=lstm_hidden, lstm_dropout=dropout
        )
        self.classifier = ClassifierHead(
            input_dim=2 * lstm_hidden,
            hidden_dims=classifier_hidden,
            dropout=dropout,
            return_logits=classifier_return_logits,
        )

    def forward(self, esm_vecs: torch.Tensor, phys_vecs: torch.Tensor, ss_vecs: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(esm_vecs, phys_vecs, ss_vecs)
        seq_out = self.seq_block(fused)
        probs = self.classifier(seq_out)
        return probs

    @torch.no_grad()
    def fit_standardizers(self, sequences: List[str], batch_size: int = 16) -> None:
        phys, ss = [], []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            phys.extend([self.phys_extractor.extract(s, normalize=False) for s in batch])
            ss.extend([self.ss_extractor.extract(s, normalize=False) for s in batch])
        import numpy as np
        phys_np = np.stack(phys, axis=0).astype("float32")
        ss_np = np.stack(ss, axis=0).astype("float32")
        self.phys_standardizer.fit(phys_np)
        self.ss_standardizer.fit(ss_np)
        self._std_fitted = True

    @torch.no_grad()
    def encode_sequences_to_features(self, sequences: List[str]) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        esm_vecs = self.embedder.encode(sequences, batch_size=8, pool="mean", sanitize=True, progress=False)  # (N, H)
        import numpy as np
        phys_list = [self.phys_extractor.extract(s, normalize=False) for s in sequences]
        ss_list = [self.ss_extractor.extract(s, normalize=False) for s in sequences]
        phys = np.stack(phys_list, axis=0).astype(np.float32)
        ss = np.stack(ss_list, axis=0).astype(np.float32)
        if self._std_fitted:
            phys = self.phys_standardizer.transform(phys)
            ss = self.ss_standardizer.transform(ss)
        return torch.from_numpy(esm_vecs.numpy()), torch.from_numpy(phys), torch.from_numpy(ss)

    @torch.no_grad()
    def predict_on_sequences(self, sequences: List[str]) -> torch.Tensor:
        """Return probability scores in [0, 1] for each sequence."""
        self.eval()
        esm_vecs, phys_vecs, ss_vecs = self.encode_sequences_to_features(sequences)
        out = self.forward(esm_vecs, phys_vecs, ss_vecs)
        # Apply sigmoid if classifier returns raw logits
        if self.classifier.return_logits:
            out = torch.sigmoid(out)
        return out