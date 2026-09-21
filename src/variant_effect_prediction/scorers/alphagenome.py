"""AlphaGenomeVariantScorer.

Default `context_len` is 131_072 bp per the alphagenome-pytorch README. The
underlying model supports variable input length up to ~1M bp; longer inputs can
be configured by passing `wrapper_kwargs={'context_len': N}`.
"""

from __future__ import annotations

import torch

from variant_effect_prediction.references import RefGenome
from variant_effect_prediction.scorers._many_tracks import _ManyTracksVariantScorer
from variant_effect_prediction.wrappers.alphagenome import (
    AlphaGenomeSummedTrack,
    ALPHAGENOME_DEFAULT_SEQ_LEN,
)


class AlphaGenomeVariantScorer(_ManyTracksVariantScorer):
    name = "alphagenome"
    context_len = ALPHAGENOME_DEFAULT_SEQ_LEN
    wrapper_cls = AlphaGenomeSummedTrack

    def __init__(
        self,
        *,
        model,
        track_idx,
        ref_genome: RefGenome,
        eval_window_len: int = 1000,
        batch_size: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        wrapper_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            model=model,
            track_idx=track_idx,
            ref_genome=ref_genome,
            eval_window_len=eval_window_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            wrapper_kwargs=wrapper_kwargs,
        )
