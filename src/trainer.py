from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from datasets import TripleFeatureDataset
from pipeline import MultiModalAFPpred


def precompute_with_pipeline(
    model: MultiModalAFPpred, sequences: list
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Use pipeline's embedder + physchem + ss to precompute tri-modal features on CPU.
    """
    esm_vecs, phys_vecs, ss_vecs = model.encode_sequences_to_features(sequences)
    return esm_vecs, phys_vecs, ss_vecs


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
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def fit_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            # Accept 4-tuple (esm, phys, ss, y)
            esm_vecs, phys_vecs, ss_vecs, labels = batch
            esm_vecs = esm_vecs.to(self.device, non_blocking=True)
            phys_vecs = phys_vecs.to(self.device, non_blocking=True)
            ss_vecs = ss_vecs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(esm_vecs, phys_vecs, ss_vecs).squeeze(-1)
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
            esm_vecs, phys_vecs, ss_vecs, labels = batch
            esm_vecs = esm_vecs.to(self.device, non_blocking=True)
            phys_vecs = phys_vecs.to(self.device, non_blocking=True)
            ss_vecs = ss_vecs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(esm_vecs, phys_vecs, ss_vecs).squeeze(-1)
                loss = self.criterion(logits, labels)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total_loss += float(loss.item()) * labels.size(0)
            n += labels.size(0)
        acc = correct / max(1, n)
        return total_loss / max(1, n), acc

    def fit(self, train_ds: TripleFeatureDataset, val_ds: TripleFeatureDataset, epochs: int = 3):
        pin = self.device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=pin)
        for ep in range(1, epochs + 1):
            train_loss = self.fit_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            print(f"[Epoch {ep}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")