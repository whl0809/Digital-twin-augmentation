"""Loss terms for the VAE objectives."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

__all__ = ["kl_divergence", "gaussian_elbo", "mixed_elbo"]


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)), summed over dimensions, averaged over the batch."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)


def gaussian_elbo(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1e-5,
) -> torch.Tensor:
    """Negative ELBO for the all-continuous :class:`~dtaug.models.VAE`.

    Reconstruction is a summed squared error, matching the original notebooks'
    ``MSELoss(reduction="sum")``; ``kl_weight`` is correspondingly small.
    """
    recon = F.mse_loss(reconstruction, target, reduction="sum") / target.size(0)
    return recon + kl_weight * kl_divergence(mu, log_var)


def mixed_elbo(
    recon_continuous: torch.Tensor,
    target_continuous: torch.Tensor,
    recon_categoricals: Sequence[torch.Tensor],
    target_categoricals: Sequence[torch.Tensor],
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1e-3,
) -> torch.Tensor:
    """Negative ELBO for :class:`~dtaug.models.MixedVAE`.

    Mean squared error on the continuous block plus one cross-entropy per
    categorical head, so a rare category is not drowned out by the (much
    larger) continuous reconstruction term.
    """
    loss = F.mse_loss(recon_continuous, target_continuous, reduction="mean")
    for prediction, target in zip(
        recon_categoricals, target_categoricals, strict=True
    ):
        loss = loss + F.cross_entropy(
            prediction, target.argmax(dim=1), reduction="mean"
        )
    return loss + kl_weight * kl_divergence(mu, log_var)
