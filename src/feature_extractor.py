"""
ESM-2 sequence-level embedding extractor.

- Loads ESM-2 from a local directory (e.g., ./ESM2)
- Provides batch encode with mean/cls pooling
- Returns CPU float32 tensor of shape (N, hidden_size)
"""
from typing import List, Literal, Optional

import torch
from transformers import AutoTokenizer, AutoModel


ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZOU")


def _sanitize_sequence(seq: str) -> str:
    """
    Uppercase the sequence and replace out-of-vocab characters with 'X'.
    """
    seq = (seq or "").upper()
    return "".join(ch if ch in ALLOWED_AA else "X" for ch in seq)


class ESM2Embedder:
    def __init__(
        self,
        model_dir: str = "./ESM2",
        device: Optional[str] = None,
        max_length: int = 1024,
        use_half: bool = False,
        local_files_only: bool = True,
    ) -> None:
        """
        Args:
            model_dir: Local directory of ESM-2 (contains config.json, pytorch_model.bin, tokenizer files).
            device: 'cuda', 'cpu' or None for auto-detect.
            max_length: Max tokens (including special tokens). Short peptides (<50) are safe.
            use_half: Use float16 on CUDA for speed/memory.
            local_files_only: Only load from local files.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = max_length

        dtype = torch.float16 if (use_half and self.device.type == "cuda") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=local_files_only)
        # Avoid remote code; ESM is supported natively by transformers
        self.model = AutoModel.from_pretrained(
            model_dir,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = getattr(self.model.config, "hidden_size", None)

        # Cache special token ids
        self.cls_id = getattr(self.tokenizer, "cls_token_id", None)
        self.eos_id = getattr(self.tokenizer, "eos_token_id", None)
        self.pad_id = getattr(self.tokenizer, "pad_token_id", None)

    @torch.no_grad()
    def encode(
        self,
        sequences: List[str],
        batch_size: int = 16,
        pool: Literal["mean", "cls"] = "mean",
        sanitize: bool = True,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Encode sequences into sequence-level embeddings.

        Args:
            sequences: List of amino-acid sequences (string of capital letters).
            batch_size: Batch size.
            pool: 'mean' (exclude special tokens) or 'cls'.
            sanitize: Replace invalid chars with 'X'.
            progress: If True, prints simple progress info.

        Returns:
            Tensor on CPU: shape (N, hidden_size)
        """
        if sanitize:
            sequences = [_sanitize_sequence(s) for s in sequences]

        outputs: List[torch.Tensor] = []
        total = len(sequences)
        for i in range(0, total, batch_size):
            chunk = sequences[i : i + batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc["attention_mask"].to(self.device)

            model_out = self.model(input_ids=input_ids, attention_mask=attn_mask)
            hidden = model_out.last_hidden_state  # [B, L, H]

            if pool == "cls":
                # Take [CLS] token (usually at position 0)
                pooled = hidden[:, 0, :]
            else:
                # Exclude special tokens: [CLS], [EOS], [PAD]
                include = attn_mask.bool()
                if self.cls_id is not None:
                    include = include & (input_ids != self.cls_id)
                if self.eos_id is not None:
                    include = include & (input_ids != self.eos_id)
                if self.pad_id is not None:
                    include = include & (input_ids != self.pad_id)

                include_f = include.float()  # [B, L]
                denom = include_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # avoid /0
                summed = (hidden * include_f.unsqueeze(-1)).sum(dim=1)  # [B, H]
                pooled = summed / denom  # [B, H]

            outputs.append(pooled.detach().to("cpu", dtype=torch.float32))

            if progress:
                print(f"[ESM2] Encoded {min(i + batch_size, total)}/{total}")

        return torch.cat(outputs, dim=0)

    def encode_one(self, sequence: str, pool: Literal["mean", "cls"] = "mean") -> torch.Tensor:
        """
        Convenience wrapper for a single sequence. Returns shape (hidden_size,)
        """
        vec = self.encode([sequence], batch_size=1, pool=pool, sanitize=True)
        return vec[0]