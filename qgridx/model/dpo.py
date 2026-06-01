"""DPO (Direct Preference Optimization) training step.

Updates the decoder to prefer circuits that produce lower-cost candidate
plans.  The reference policy is a FROZEN copy of the model at initialization
(consistent with Conditional-GQE paper, arXiv 2501.16986).

# TODO(scientific-review): the DPO-through-circuit-scores loop is the piece
# most likely to need scientific iteration.  This is a draft implementation.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qgridx.config import DPOConfig
from qgridx.model.decoder_transformer import PCEBrickworkDecoder


class DPOTrainer:
    """Trains the decoder via DPO given ranked circuit pairs.

    Args:
        policy:    The active :class:`PCEBrickworkDecoder` (will be updated).
        cfg:       :class:`DPOConfig`.
        device:    Torch device string.
    """

    def __init__(
        self,
        policy: PCEBrickworkDecoder,
        cfg: DPOConfig,
        device: str = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device

        # Frozen reference policy (copy of initialization)
        self.ref_policy = copy.deepcopy(policy).to(device)
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self.ref_policy.eval()

        self.optimizer = torch.optim.AdamW(
            policy.parameters(), lr=cfg.lr
        )

    def _log_prob(
        self,
        model: PCEBrickworkDecoder,
        token_ids: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute sum of log-probabilities for a token sequence.

        Args:
            model:     The model to evaluate.
            token_ids: (B, T) token sequence including BOS.
            context:   (B, n_embd) context or None.

        Returns:
            (B,) sum of log-probs over the non-padding tokens.
        """
        logits = model(token_ids[:, :-1], context=context)  # (B, T-1, V)
        targets = token_ids[:, 1:]  # (B, T-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log-probs at the target tokens
        token_log_probs = log_probs.gather(
            2, targets.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)
        # Mask padding
        pad_id = model.tokenizer.pad_id
        mask = (targets != pad_id).float()
        return (token_log_probs * mask).sum(dim=-1)  # (B,)

    def step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> float:
        """Perform one DPO gradient step.

        Args:
            chosen_ids:  (B, T) token ids for preferred (lower-cost) circuits.
            rejected_ids:(B, T) token ids for dispreferred (higher-cost) circuits.
            context:     (B, n_embd) context vectors, or None.

        Returns:
            Scalar DPO loss value.
        """
        self.policy.train()
        chosen_ids = chosen_ids.to(self.device)
        rejected_ids = rejected_ids.to(self.device)
        if context is not None:
            context = context.to(self.device)

        # Policy log-probs
        pi_chosen = self._log_prob(self.policy, chosen_ids, context)
        pi_rejected = self._log_prob(self.policy, rejected_ids, context)

        # Reference log-probs (no grad)
        with torch.no_grad():
            ref_chosen = self._log_prob(self.ref_policy, chosen_ids, context)
            ref_rejected = self._log_prob(self.ref_policy, rejected_ids, context)

        # DPO loss
        beta = self.cfg.beta
        log_ratio_chosen = pi_chosen - ref_chosen
        log_ratio_rejected = pi_rejected - ref_rejected
        loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
