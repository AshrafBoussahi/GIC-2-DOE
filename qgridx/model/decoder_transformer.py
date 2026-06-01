"""Decoder-only Transformer for PCE-brickwork circuit generation.

Architecture follows the gqco reference implementation (MIT licensed):
  - Causal (decoder-only) Transformer with masked self-attention.
  - Context vector from the GNN encoder is injected as a prefix embedding.
  - Generation masking enforces the brickwork grammar at every step.

Attribution: Architecture adapted from
  https://github.com/shunyaist/generative-quantum-combinatorial-optimization
  (MIT License).

# TODO(scientific-review): DPO-through-circuit-scores is the piece most likely
# to need scientific iteration; treat the first implementation as a draft.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qgridx.config import DecoderConfig
from qgridx.model.tokenizer import CircuitTokenizer


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, max_len: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        # Causal mask
        mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in qkv]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, max_len: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_len)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class PCEBrickworkDecoder(nn.Module):
    """Decoder-only Transformer generating PCE-brickwork circuit token sequences.

    The context vector from the GNN encoder is projected to the embedding
    dimension and prepended as the first token of the sequence.

    Args:
        cfg:       :class:`DecoderConfig`.
        tokenizer: :class:`CircuitTokenizer` for this problem instance.
    """

    def __init__(self, cfg: DecoderConfig, tokenizer: CircuitTokenizer) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        V = tokenizer.vocab_size

        self.tok_emb = nn.Embedding(V, cfg.n_embd, padding_idx=tokenizer.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len + 1, cfg.n_embd)  # +1 for context prefix
        self.ctx_proj = nn.Linear(cfg.n_embd, cfg.n_embd)  # GNN context → embd
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.n_embd, cfg.n_head, cfg.max_len + 1)
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, V, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits for next-token prediction.

        Args:
            token_ids: (B, T) tensor of token indices.
            context:   (B, n_embd) context vector from GNN encoder.

        Returns:
            (B, T, V) logits.
        """
        B, T = token_ids.shape
        tok = self.tok_emb(token_ids)  # (B, T, C)

        if context is not None:
            ctx_emb = self.ctx_proj(context).unsqueeze(1)  # (B, 1, C)
            x = torch.cat([ctx_emb, tok], dim=1)  # (B, T+1, C)
        else:
            x = tok

        T_full = x.shape[1]
        pos = torch.arange(T_full, device=x.device)
        x = x + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T_full, V)

        # Remove the prepended context position from the output
        if context is not None:
            logits = logits[:, 1:, :]  # (B, T, V)

        return logits

    @torch.no_grad()
    def generate(
        self,
        context: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
        device: str = "cpu",
        grammar_mask: bool = True,
    ) -> list[int]:
        """Autoregressively generate a single token sequence.

        Args:
            context:      (n_embd,) context vector (no batch dim).
            max_len:      Maximum sequence length.
            device:       Torch device.
            grammar_mask: Whether to apply the brickwork grammar mask.

        Returns:
            List of token ids (not including BOS).
        """
        self.eval()
        self.to(device)
        max_len = max_len or self.cfg.max_len
        ctx_batch = context.unsqueeze(0).to(device) if context is not None else None

        generated = [self.tokenizer.bos_id]
        for _ in range(max_len):
            ids = torch.tensor([generated], dtype=torch.long, device=device)
            logits = self.forward(ids, context=ctx_batch)
            next_logits = logits[0, -1, :]  # (V,)

            if grammar_mask:
                mask = self.tokenizer.grammar_mask(generated)
                mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
                next_logits = next_logits.masked_fill(~mask_t, float("-inf"))

            next_id = int(torch.argmax(next_logits).item())
            generated.append(next_id)

            if next_id == self.tokenizer.end_id:
                break

        return generated[1:]  # strip BOS
