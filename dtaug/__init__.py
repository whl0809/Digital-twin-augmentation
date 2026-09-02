"""Digital-twin augmentation: VAE-based synthetic patient generation.

Trains variational autoencoders on cardiovascular digital-twin parameter sets
and samples synthetic ("fake") patients from the learned latent space.
"""

from dtaug.data import PreparedData, prepare
from dtaug.models import VAE, AutoEncoder, ConditionalVAE, MixedVAE
from dtaug.sampling import sample_mixed_vae, sample_vae
from dtaug.schema import ALL149, CORE35, MIXED149, DatasetSchema
from dtaug.train import train_mixed_vae, train_vae

__all__ = [
    "DatasetSchema",
    "CORE35",
    "ALL149",
    "MIXED149",
    "PreparedData",
    "prepare",
    "AutoEncoder",
    "VAE",
    "MixedVAE",
    "ConditionalVAE",
    "train_vae",
    "train_mixed_vae",
    "sample_vae",
    "sample_mixed_vae",
]
