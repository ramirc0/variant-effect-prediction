"""Shared scorer body for many-tracks models (Enformer, Borzoi, AlphaGenome).

Each many-tracks scorer wraps a model via a tangermeme-compatible
ManyTracksWrapperBase subclass, then runs forward-only batched prediction over
ref and alt sequences and returns (N, n_tracks, 1) log counts.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from tangermeme.predict import predict

from variant_effect_prediction.scoring import one_hot_batch
from variant_effect_prediction.scorers.base import VariantScorer
from variant_effect_prediction.wrappers.base import ManyTracksWrapperBase


# Cap the host-side one-hot footprint. The full-set one-hot is (N, 4, context_len)
# float32 *per allele* (X1 and X2 alive at once); Borzoi's 524 kb context makes that
# ~130 GB for a ~7.9k-variant QTL set, which OOM-kills a 125 GB SLURM job. So we
# one-hot + predict in host-chunks sized to this budget and free between them — the
# per-sequence predictions are independent, so the concatenated result is identical.
_HOST_ONEHOT_BUDGET_BYTES = 8 * 1024**3  # ~8 GiB per allele chunk tensor


class _ManyTracksVariantScorer(VariantScorer):
    wrapper_cls: ClassVar[type[ManyTracksWrapperBase]]
    # Many-tracks heads (Enformer/Borzoi/AlphaGenome) emit non-negative coverage,
    # so JSD profiles are sum-normalized (BPNet-like logit heads use softmax).
    profile_normalization: ClassVar[str] = "sum"

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        track_idx,
        ref_genome,
        eval_window_len: int = 1000,
        batch_size: int = 8,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        wrapper_kwargs: dict | None = None,
        host_chunk_size: int | None = None,
    ) -> None:
        super().__init__(
            ref_genome=ref_genome,
            eval_window_len=eval_window_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        wkw = wrapper_kwargs or {}
        self.wrapper = self.wrapper_cls(
            model, track_idx, eval_window_len=eval_window_len, **wkw
        )
        self.wrapper.eval()
        self.wrapper.to(device)

        # The wrapper is the authority on input length (AlphaGenome takes it as a ctor
        # arg), so adopt it before sizing chunks: fetch + chunking must agree.
        self.context_len = getattr(self.wrapper, "context_len", self.context_len)

        # How many variants to one-hot on the host at a time (bounds host RAM).
        # Derived from the context length unless the caller overrides it.
        if host_chunk_size is not None:
            self.host_chunk_size = max(1, int(host_chunk_size))
        else:
            bytes_per_seq = 4 * self.context_len * 4  # (4 channels, float32)
            self.host_chunk_size = max(1, _HOST_ONEHOT_BUDGET_BYTES // bytes_per_seq)

    def _predict_alleles(self, allele1_seqs, allele2_seqs):
        n = len(allele1_seqs)
        chunk = self.host_chunk_size
        y1s, y2s, p1s, p2s = [], [], [], []
        with torch.no_grad():
            for i in range(0, n, chunk):
                X1 = one_hot_batch(allele1_seqs[i : i + chunk])
                X2 = one_hot_batch(allele2_seqs[i : i + chunk])
                prof1, y1 = predict(
                    self.wrapper, X1, batch_size=self.batch_size,
                    device=self.device, dtype=self.dtype,
                )
                prof2, y2 = predict(
                    self.wrapper, X2, batch_size=self.batch_size,
                    device=self.device, dtype=self.dtype,
                )
                y1s.append(y1.cpu())
                y2s.append(y2.cpu())
                p1s.append(prof1.cpu())
                p2s.append(prof2.cpu())
                del X1, X2, prof1, prof2, y1, y2
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        y1 = torch.cat(y1s)
        y2 = torch.cat(y2s)
        prof1 = torch.cat(p1s)
        prof2 = torch.cat(p2s)
        # counts: (N, n_tracks) → (N, n_tracks, 1) [ensemble dim = tracks].
        # profile: (N, n_center_bins) → (N, 1, n_center_bins) [single model, no folds].
        return (
            y1.unsqueeze(-1),
            y2.unsqueeze(-1),
            prof1.unsqueeze(1),
            prof2.unsqueeze(1),
        )
