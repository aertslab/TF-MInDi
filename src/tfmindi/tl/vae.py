"""
Variational Autoencoder (VAE) for seqlet pattern dimensionality reduction.

This module provides a drop-in alternative to PCA for compressing high-dimensional
seqlet contribution score matrices into a lower-dimensional latent space. The VAE
learns a non-linear manifold and can capture structure that PCA misses, particularly
useful when seqlet patterns have complex, non-Gaussian distributions.

Typical usage (called automatically via ``cluster_seqlets(..., reduction="vae")``):

    from tfmindi.tl.vae import fit_vae_latents
    Z = fit_vae_latents(X, latent_dim=10)

Architecture
------------
- Encoder: MLP (ReLU + Dropout) → μ, log σ² heads
- Reparameterisation trick: z = μ + ε·σ,  ε ~ N(0,I)
- Decoder: MLP (ReLU + Dropout) → reconstruction
- Loss:  MSE reconstruction  +  β · KL divergence  (β-VAE formulation)
- Embedding returned: posterior mean μ (deterministic, used at inference time)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_feature_mean_std(
    X: np.ndarray | sp.spmatrix,
    n_sample: int = 20_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-feature mean and std from a (possibly large) matrix.

    Parameters
    ----------
    X
        Input matrix of shape ``(n_obs, n_features)``.  Dense or CSR sparse.
    n_sample
        Number of rows to subsample for estimation.  When ``n_obs <= n_sample``
        all rows are used.
    seed
        Random seed for reproducible subsampling.

    Returns
    -------
    mean, std
        Float32 arrays of shape ``(n_features,)``.  Features with
        ``std < 1e-6`` are assigned ``std = 1.0`` to avoid division-by-zero.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(n_sample, n), replace=False)

    subset = X[idx].toarray().astype(np.float32) if sp.issparse(X) else np.asarray(X[idx], dtype=np.float32)

    mean = subset.mean(axis=0)
    std = subset.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _make_dataset(
    X: np.ndarray | sp.spmatrix,
    mean: np.ndarray,
    std: np.ndarray,
):
    """Return a ``torch.utils.data.Dataset`` that streams standardised rows.

    Rows are standardised on-the-fly in ``__getitem__`` to avoid materialising
    the full dense matrix in memory.
    """
    from torch.utils.data import Dataset

    class _SeqletDataset(Dataset):
        def __init__(self, X, mean, std):
            self.X = X
            self.mean = mean
            self.std = std
            self._sparse = sp.issparse(X)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            import torch

            row = (
                self.X[idx].toarray().ravel().astype(np.float32)
                if self._sparse
                else np.asarray(self.X[idx], dtype=np.float32).ravel()
            )
            return torch.from_numpy((row - self.mean) / self.std)

    return _SeqletDataset(X, mean, std)


# ---------------------------------------------------------------------------
# VAE model
# ---------------------------------------------------------------------------


def _build_mlp(in_dim: int, hidden: int, n_layers: int, out_dim: int, dropout: float = 0.1):
    """Stack ``n_layers`` Linear->ReLU->Dropout blocks followed by a Linear output layer."""
    import torch.nn as nn

    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _build_vae(in_dim: int, latent_dim: int, hidden: int, n_layers: int, dropout: float = 0.1):
    """Construct and return a VAE ``nn.Module``.

    The class is defined inside this factory function so that ``torch`` is only
    imported when VAE reduction is actually requested.
    """
    import torch
    import torch.nn as nn

    class _VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = _build_mlp(in_dim, hidden, n_layers, hidden, dropout)
            self.mu_head = nn.Linear(hidden, latent_dim)
            self.logvar_head = nn.Linear(hidden, latent_dim)
            self.dec = _build_mlp(latent_dim, hidden, n_layers, in_dim, dropout)

        def _reparam(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def forward(self, x):
            h = self.enc(x)
            mu = self.mu_head(h)
            logvar = self.logvar_head(h)
            z = self._reparam(mu, logvar)
            return self.dec(z), mu, logvar

    return _VAE()


def _vae_loss(x, xhat, mu, logvar, beta: float):
    """beta-VAE loss: MSE reconstruction + beta * KL divergence.

    KL is summed over the latent dimension then averaged over the batch, which
    keeps its scale independent of ``latent_dim`` and makes ``beta`` comparable
    across runs with different latent sizes.

    Logvar is clamped to ``[-4, 15]`` before exponentiation to prevent numerical
    overflow early in training.
    """
    import torch.nn.functional as F

    recon = F.mse_loss(xhat, x, reduction="mean")
    logvar = logvar.clamp(-4, 15)
    # sum over latent dim, mean over batch -> scale is independent of latent_dim
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
    return recon + beta * kl, recon, kl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_vae_latents(
    X: np.ndarray | sp.spmatrix,
    latent_dim: int = 10,
    *,
    hidden: int = 512,
    n_layers: int = 2,
    beta: float = 0.1,
    lr: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 4096,
    num_workers: int = 0,
    use_amp: bool = True,
    dropout: float = 0.1,
    n_sample_stats: int = 20_000,
    seed: int = 0,
    device: Literal["cpu", "cuda", "auto"] = "auto",
    verbose: bool = True,
) -> np.ndarray:
    """Train a VAE on seqlet contribution scores and return the latent embedding.

    The VAE encodes each seqlet's flattened contribution score vector into a
    ``latent_dim``-dimensional Gaussian posterior.  The posterior mean **mu** is
    returned as the deterministic embedding (analogous to PCA coordinates).

    Parameters
    ----------
    X
        Input matrix of shape ``(n_seqlets, n_features)``.  Accepts dense
        ``np.ndarray`` or any SciPy sparse matrix (CSR recommended for speed).
    latent_dim
        Number of latent dimensions.  Good starting values: 10-15 for typical
        TF-MINDI runs; increase if Leiden clustering is under-resolved.
    hidden
        Width of each hidden layer in the encoder/decoder MLP.
    n_layers
        Number of MLP blocks in encoder and decoder (each block is Linear->ReLU->Dropout).
    beta
        Weight of the KL divergence term (beta-VAE).  Lower values (0.1) give
        more reconstruction-faithful embeddings; higher values (1-4) push
        the posterior closer to a unit Gaussian and improve disentanglement.
    lr
        AdamW learning rate.
    epochs
        Training epochs.  50 is usually sufficient; increase to 100-200 for
        large datasets or if the loss is still decreasing.
    batch_size
        Mini-batch size.  Reduce if you hit GPU out-of-memory errors.
    num_workers
        DataLoader worker processes.  Set to 0 when running in notebooks or
        environments where multiprocessing causes issues.
    use_amp
        Enable automatic mixed precision (AMP) on CUDA.  Ignored on CPU.
    dropout
        Dropout rate applied after each hidden layer.
    n_sample_stats
        Number of rows used to estimate per-feature mean/std for
        standardisation.  The full matrix is **not** loaded into memory.
    seed
        Random seed passed to both numpy (for stats subsampling) and
        ``torch.manual_seed`` (for weight initialisation and dropout), so
        results are fully reproducible across runs.
    device
        ``"auto"`` selects CUDA if available, otherwise CPU.
        Pass ``"cpu"`` or ``"cuda"`` to override.
    verbose
        Print per-epoch loss breakdown (total, reconstruction, KL).

    Returns
    -------
    Z
        Float32 array of shape ``(n_seqlets, latent_dim)`` - the VAE posterior
        means, ready to be stored in ``adata.obsm["X_vae"]``.

    Raises
    ------
    ImportError
        If PyTorch is not installed.  Install with ``pip install tfmindi[vae]``
        or ``pip install torch``.

    Notes
    -----
    * Feature standardisation is applied on-the-fly inside the Dataset; the
      original matrix ``X`` is never modified.
    * The returned embedding is **deterministic** (uses mu, not a sample z).
    * KL is summed over the latent dimension then averaged over the batch, so
      the effective weight of ``beta`` does not change when ``latent_dim`` is varied.

    Examples
    --------
    >>> Z = fit_vae_latents(adata.X, latent_dim=10, epochs=50)
    >>> adata.obsm["X_vae"] = Z
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for VAE-based dimensionality reduction. "
            "Install it with:  pip install tfmindi[vae]\n"
            "See https://pytorch.org for platform-specific instructions."
        ) from e

    # ------------------------------------------------------------------ setup
    if device == "auto":
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = device

    # Seed both numpy (stats subsampling) and torch (weights + dropout)
    torch.manual_seed(seed)

    print(f"Training VAE on {_device} (latent_dim={latent_dim}, epochs={epochs}, beta={beta})...")

    # ----------------------------------------------------------------- stats
    mean, std = _compute_feature_mean_std(X, n_sample=n_sample_stats, seed=seed)
    dataset = _make_dataset(X, mean, std)

    _pin = _device == "cuda"
    _persist = num_workers > 0

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin,
        persistent_workers=_persist,
    )

    # ----------------------------------------------------------- model / opt
    in_dim = X.shape[1]
    model = _build_vae(in_dim, latent_dim, hidden, n_layers, dropout).to(_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # _amp_enabled is False on CPU so autocast and GradScaler both no-op cleanly
    _amp_enabled = use_amp and _device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled)

    # ---------------------------------------------------------- train loop
    model.train()
    for epoch in range(1, epochs + 1):
        tot_loss = tot_recon = tot_kl = 0.0
        n_batches = 0

        for xb in train_loader:
            xb = xb.to(_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(_device, enabled=_amp_enabled):
                xhat, mu, logvar = model(xb)
                loss, recon, kl = _vae_loss(xb, xhat, mu, logvar, beta)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tot_loss += loss.item()
            tot_recon += recon.item()
            tot_kl += kl.item()
            n_batches += 1

        if verbose:
            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"loss={tot_loss / n_batches:.4f}  "
                f"recon={tot_recon / n_batches:.4f}  "
                f"kl={tot_kl / n_batches:.4f}"
            )

    # ----------------------------------------------------------- embed (mu)
    model.eval()
    embed_loader = DataLoader(
        dataset,
        batch_size=8192,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin,
    )

    Z_parts: list[np.ndarray] = []
    with torch.no_grad():
        for xb in embed_loader:
            xb = xb.to(_device, non_blocking=True)
            h = model.enc(xb)
            mu = model.mu_head(h)
            Z_parts.append(mu.cpu().numpy())

    Z = np.vstack(Z_parts).astype(np.float32)
    print(f"VAE embedding complete. Shape: {Z.shape}")
    return Z