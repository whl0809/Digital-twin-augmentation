"""Autoencoder / VAE architectures.

Three decoders, all sharing the same encoder trunk:

* :class:`AutoEncoder` -- plain encoder/decoder, all-continuous output.
* :class:`VAE` -- adds the reparameterisation trick on top of ``AutoEncoder``.
* :class:`MixedVAE` -- one Gaussian head for the continuous columns plus a
  Gumbel-softmax head per categorical column, so discrete clinical fields are
  sampled as categories instead of being rounded after the fact.
* :class:`ConditionalVAE` -- :class:`VAE` with the latent code shifted by a
  projection of a class label.

Layer widths (``input_size -> 256 -> latent``, ``latent -> 256 -> 128 -> out``)
are carried over unchanged from the original notebooks.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AutoEncoder", "VAE", "MixedVAE", "ConditionalVAE", "reparameterize"]

DEFAULT_LATENT_DIM = 16


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Sample from N(mu, exp(log_var)) so the draw stays differentiable."""
    std = torch.exp(0.5 * log_var)
    return mu + torch.randn_like(std) * std


class AutoEncoder(nn.Module):
    """Deterministic autoencoder over a fully continuous feature vector."""

    def __init__(self, input_size: int, latent_dim: int = DEFAULT_LATENT_DIM):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


class VAE(AutoEncoder):
    """Variational autoencoder with a diagonal-Gaussian latent."""

    def __init__(self, input_size: int, latent_dim: int = DEFAULT_LATENT_DIM):
        super().__init__(input_size, latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = self.mu(encoded), self.log_var(encoded)
        z = reparameterize(mu, log_var)
        return encoded, self.decoder(z), mu, log_var

    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        """Draw ``num_samples`` vectors from the prior and decode them."""
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)


class MixedVAE(nn.Module):
    """VAE with separate continuous and categorical decoder heads.

    The model input is the continuous block concatenated with the one-hot
    encoding of every categorical column, so ``input_size`` is
    ``continuous_dim + sum(categorical_dims)``.

    Args:
        continuous_dim: Number of continuous columns.
        categorical_dims: Cardinality of each categorical column, in the order
            the heads and the one-hot blocks are concatenated.
        latent_dim: Width of the latent code.
    """

    def __init__(
        self,
        continuous_dim: int,
        categorical_dims: Sequence[int],
        latent_dim: int = DEFAULT_LATENT_DIM,
    ):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_dims = list(categorical_dims)
        self.latent_dim = latent_dim
        self.input_size = continuous_dim + sum(self.categorical_dims)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.head_continuous = nn.Linear(128, continuous_dim)
        self.head_categorical = nn.ModuleList(
            [nn.Linear(128, dim) for dim in self.categorical_dims]
        )

    def decode(
        self, z: torch.Tensor, temperature: float, hard: bool
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Decode a latent code into continuous values and categorical samples.

        Args:
            z: Latent codes, shape ``(batch, latent_dim)``.
            temperature: Gumbel-softmax temperature; lower is closer to one-hot.
            hard: Straight-through one-hot output.  Use ``False`` while
                training (keeps gradients smooth) and ``True`` when sampling.

        Returns:
            The continuous block and one tensor per categorical head.  Each
            categorical tensor holds *logits-derived samples*, not raw logits.
        """
        h = F.relu(self.fc2(F.relu(self.fc1(z))))
        categoricals = [
            F.gumbel_softmax(head(h), tau=temperature, hard=hard)
            for head in self.head_categorical
        ]
        return self.head_continuous(h), categoricals

    def forward(self, x: torch.Tensor, temperature: float = 0.5):
        encoded = self.encoder(x)
        mu, log_var = self.mu(encoded), self.log_var(encoded)
        z = reparameterize(mu, log_var)
        continuous, categoricals = self.decode(z, temperature, hard=False)
        return encoded, continuous, categoricals, mu, log_var

    @torch.no_grad()
    def sample(self, num_samples: int, temperature: float = 0.1):
        """Draw from the prior; categorical heads return hard one-hot vectors."""
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, temperature, hard=True)


class ConditionalVAE(VAE):
    """VAE whose latent code is shifted by a projection of a class label.

    Kept from the original notebooks; not used by the shipped scripts.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
    ):
        super().__init__(input_size, latent_dim)
        self.label_projector = nn.Sequential(
            nn.Linear(num_classes, latent_dim),
            nn.ReLU(),
        )

    def condition_on_label(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return z + self.label_projector(y.float())

    def forward(self, x, y):
        encoded = self.encoder(x)
        mu, log_var = self.mu(encoded), self.log_var(encoded)
        z = reparameterize(mu, log_var)
        return encoded, self.decoder(self.condition_on_label(z, y)), mu, log_var

    @torch.no_grad()
    def sample(self, num_samples: int, y: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(self.condition_on_label(z, y))
