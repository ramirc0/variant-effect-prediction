"""Tangermeme wrapper for Enformer.

Enformer:
    - Input: (B, 196_608, 4) one-hot float OR (B, 196_608) long indices.
    - Output: dict {'human': (B, 896, 5313), 'mouse': (B, 896, 1643)}.
    - bin_size = 128 bp, central 896 bins (114.7 kb of context).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from variant_effect_prediction.wrappers.base import ManyTracksWrapperBase


ENFORMER_SEQ_LEN = 196_608


def _require_tf_gamma(enformer: nn.Module) -> None:
    from enformer_pytorch.modeling_enformer import Attention

    flags = [m.use_tf_gamma for m in enformer.modules() if isinstance(m, Attention)]
    if not all(flags):
        raise ValueError(
            "Enformer must be loaded with use_tf_gamma=True, e.g. "
            "Enformer.from_pretrained(path, use_tf_gamma=True)"
        )


class EnformerSummedTrack(ManyTracksWrapperBase):
    bin_size = 128
    n_output_bins = 896

    def __init__(
        self,
        enformer: nn.Module,
        track_idx: Sequence[int],
        eval_window_len: int = 1000,
        head: str = "human",
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(enformer, track_idx, eval_window_len, epsilon=epsilon)
        self.head = head
        _require_tf_gamma(enformer)

    def _predict_central_tracks(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] != ENFORMER_SEQ_LEN:
            raise ValueError(
                f"Enformer expects sequences of length {ENFORMER_SEQ_LEN}, got {X.shape[-1]}"
            )
        x = X.permute(0, 2, 1).contiguous()  # (B, L, 4)
        out = self.model(x, head=self.head)  # (B, 896, n_tracks)
        out = out[:, self.bin_lo : self.bin_hi, :]  # (B, n_center, n_tracks)
        return out.index_select(-1, self.track_idx)  # (B, n_center, len(track_idx))
