"""Circuit token vocabulary and grammar mask for the Transformer decoder.

The vocabulary consists of tokens that represent PCE-brickwork circuit elements:
  - ROT_qi_a<k>  — rotation Ry on qubit i at angle bucket k
  - ENT_qi_qj    — nearest-neighbour CX from qubit i to qubit j (|i-j|==1)
  - END           — end-of-circuit token
  - PAD           — padding token (index 0)
  - BOS           — beginning-of-sequence token

The grammar mask ensures the decoder cannot emit tokens that violate the
brickwork structure (e.g. long-range CX, CX between same qubit).
"""
from __future__ import annotations

import math


N_ANGLE_BUCKETS = 16


class CircuitTokenizer:
    """Token vocabulary for PCE-brickwork circuits.

    Args:
        n_qubits:  Number of qubits in the ansatz.
        topology:  Connectivity topology. Only ``"linear"`` is implemented.
    """

    def __init__(self, n_qubits: int, topology: str = "linear") -> None:
        self.n = n_qubits
        self.topology = topology
        self._build_vocab()

    def _build_vocab(self) -> None:
        tokens = ["PAD", "BOS", "END"]
        for q in range(self.n):
            for a in range(N_ANGLE_BUCKETS):
                tokens.append(f"ROT_q{q}_a{a}")
        if self.topology == "linear":
            for q in range(self.n - 1):
                tokens.append(f"ENT_q{q}_q{q+1}")
        else:
            raise NotImplementedError(f"Topology '{self.topology}' not supported.")
        self._token_to_id = {t: i for i, t in enumerate(tokens)}
        self._id_to_token = tokens

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    @property
    def pad_id(self) -> int:
        return self._token_to_id["PAD"]

    @property
    def bos_id(self) -> int:
        return self._token_to_id["BOS"]

    @property
    def end_id(self) -> int:
        return self._token_to_id["END"]

    def encode(self, token: str) -> int:
        if token not in self._token_to_id:
            raise ValueError(f"Unknown token: '{token}'. Vocab: {self._id_to_token}")
        return self._token_to_id[token]

    def decode(self, token_id: int) -> str:
        return self._id_to_token[token_id]

    def encode_sequence(self, tokens: list[str]) -> list[int]:
        return [self.encode(t) for t in tokens]

    def decode_sequence(self, ids: list[int]) -> list[str]:
        return [self.decode(i) for i in ids]

    def grammar_mask(self, prefix: list[int]) -> list[bool]:
        """Return a boolean mask over the vocabulary.

        True = token is allowed as the next token given *prefix*.
        The mask enforces brickwork grammar:
          - BOS must be first.
          - END is always allowed after at least one real token.
          - After END, only PAD is allowed.
          - PAD only allowed after END.
          - All other tokens are allowed unless END/PAD was already emitted.

        Args:
            prefix: List of token ids emitted so far (not including BOS).

        Returns:
            Boolean list of length vocab_size.
        """
        mask = [False] * self.vocab_size
        if not prefix or prefix[-1] == self.bos_id:
            # After BOS: allow any non-PAD, non-BOS token
            for i in range(self.vocab_size):
                t = self._id_to_token[i]
                if t not in ("PAD", "BOS"):
                    mask[i] = True
            return mask

        if prefix[-1] == self.end_id:
            # After END: only PAD
            mask[self.pad_id] = True
            return mask

        if self.end_id in prefix:
            # Already ended
            mask[self.pad_id] = True
            return mask

        # Mid-sequence: allow any real token + END
        for i in range(self.vocab_size):
            t = self._id_to_token[i]
            if t not in ("PAD", "BOS"):
                mask[i] = True
        return mask
