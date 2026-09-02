"""Training loops for the continuous and mixed VAEs."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset

from dtaug.losses import gaussian_elbo, mixed_elbo
from dtaug.models import VAE, MixedVAE

__all__ = ["train_vae", "train_mixed_vae"]

ProgressFn = Callable[[int, int, float], None]


def _log(epoch: int, num_epochs: int, loss: float) -> None:
    print(f"epoch {epoch:>4}/{num_epochs}  loss={loss:.4f}")


def _should_stop(loss: float, threshold: float | None) -> bool:
    return threshold is not None and loss < threshold


def train_vae(
    X: torch.Tensor,
    *,
    latent_dim: int = 16,
    learning_rate: float = 1e-3,
    num_epochs: int = 600,
    batch_size: int = 16,
    kl_weight: float = 1e-5,
    stop_below: float | None = None,
    device: torch.device | str = "cpu",
    on_epoch: ProgressFn = _log,
) -> VAE:
    """Fit an all-continuous :class:`~dtaug.models.VAE` on a scaled matrix.

    Args:
        X: Scaled feature matrix, shape ``(n_rows, n_features)``.
        stop_below: Stop early once the mean per-sample epoch loss falls below
            this value.  ``None`` (the default) runs all ``num_epochs``.

    Returns:
        The trained model, left on ``device`` and in ``eval`` mode.
    """
    device = torch.device(device)
    model = VAE(input_size=X.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, num_epochs + 1):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            _, decoded, mu, log_var = model(batch)
            loss = gaussian_elbo(decoded, batch, mu, log_var, kl_weight=kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.size(0)

        epoch_loss = total / len(loader.dataset)
        on_epoch(epoch, num_epochs, epoch_loss)
        if _should_stop(epoch_loss, stop_below):
            print(f"early stop at epoch {epoch} (loss {epoch_loss:.4f})")
            break

    return model.eval()


def train_mixed_vae(
    X_continuous: torch.Tensor,
    X_categoricals: Sequence[torch.Tensor],
    *,
    latent_dim: int = 16,
    learning_rate: float = 1e-3,
    num_epochs: int = 300,
    batch_size: int = 16,
    kl_weight: float = 1e-3,
    temperature: float = 0.5,
    stop_below: float | None = None,
    device: torch.device | str = "cpu",
    on_epoch: ProgressFn = _log,
) -> MixedVAE:
    """Fit a :class:`~dtaug.models.MixedVAE` on continuous + one-hot blocks.

    Args:
        X_continuous: Scaled continuous block, shape ``(n_rows, continuous_dim)``.
        X_categoricals: One one-hot matrix per categorical column, in the order
            the decoder's heads should follow.
        temperature: Gumbel-softmax temperature used during training.

    Returns:
        The trained model, left on ``device`` and in ``eval`` mode.
    """
    device = torch.device(device)
    dims = [c.shape[1] for c in X_categoricals]
    model = MixedVAE(X_continuous.shape[1], dims, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(X_continuous, *X_categoricals),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(1, num_epochs + 1):
        total = 0.0
        for batch in loader:
            continuous = batch[0].to(device)
            categoricals = [c.to(device) for c in batch[1:]]

            inputs = torch.cat([continuous, *categoricals], dim=1)
            _, recon_c, recon_cats, mu, log_var = model(inputs, temperature=temperature)
            loss = mixed_elbo(
                recon_c, continuous, recon_cats, categoricals,
                mu, log_var, kl_weight=kl_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * continuous.size(0)

        epoch_loss = total / len(loader.dataset)
        on_epoch(epoch, num_epochs, epoch_loss)
        if _should_stop(epoch_loss, stop_below):
            print(f"early stop at epoch {epoch} (loss {epoch_loss:.4f})")
            break

    return model.eval()
