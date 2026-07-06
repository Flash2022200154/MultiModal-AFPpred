from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from datasets import FeatureDataset
from pipeline import MultiModalAFPpred


def precompute_with_pipeline(
    model: MultiModalAFPpred, sequences: list
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use pipeline's embedder + physchem to precompute features on CPU.
    """
    esm_vecs, phys_vecs = model.encode_sequences_to_features(sequences)
    return esm_vecs, phys_vecs


class SimpleTrainer:
    def __init__(self, model: MultiModalAFPpred, lr: float = 1e-3, batch_size: int = 16, num_workers: int = 0, device: str = None, use_amp: bool = True):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.model.to(self.device)

        # We expect classifier to return logits for stable training
        if not getattr(self.model.classifier, "return_logits", False):
            print("[WARN] ClassifierHead.return_logits is False; enabling logits improves training stability.")

        self.criterion = nn.BCEWithLogitsLoss()  # works on logits
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def fit_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            # Accept 3-tuple (esm, phys, y) or 4-tuple (esm, phys, ss, y)
            if len(batch) == 3:
                esm_vecs, phys_vecs, labels = batch
                ss_vecs = None
            else:
                esm_vecs, phys_vecs, ss_vecs, labels = batch
            esm_vecs = esm_vecs.to(self.device, non_blocking=True)
            phys_vecs = phys_vecs.to(self.device, non_blocking=True)
            if ss_vecs is not None:
                ss_vecs = ss_vecs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if ss_vecs is None:
                    logits = self.model(esm_vecs, phys_vecs)
                else:
                    logits = self.model(esm_vecs, phys_vecs, ss_vecs)
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                loss = self.criterion(logits, labels)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += float(loss.item()) * labels.size(0)
            n += labels.size(0)
        return total_loss / max(1, n)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        n = 0
        correct = 0
        for batch in loader:
            if len(batch) == 3:
                esm_vecs, phys_vecs, labels = batch
                ss_vecs = None
            else:
                esm_vecs, phys_vecs, ss_vecs, labels = batch
            esm_vecs = esm_vecs.to(self.device, non_blocking=True)
            phys_vecs = phys_vecs.to(self.device, non_blocking=True)
            if ss_vecs is not None:
                ss_vecs = ss_vecs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if ss_vecs is None:
                    logits = self.model(esm_vecs, phys_vecs)
                else:
                    logits = self.model(esm_vecs, phys_vecs, ss_vecs)
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                loss = self.criterion(logits, labels)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total_loss += float(loss.item()) * labels.size(0)
            n += labels.size(0)
        acc = correct / max(1, n)
        return total_loss / max(1, n), acc

    def fit(self, train_ds: FeatureDataset, val_ds: FeatureDataset, epochs: int = 3):
        pin = self.device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=pin)
        for ep in range(1, epochs + 1):
            train_loss = self.fit_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            print(f"[Epoch {ep}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")