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


def _vae_loss(x, xhat, mu, logvar, beta: float, free_bits: float = 0.0):
    """beta-VAE loss: MSE reconstruction + beta * KL divergence.

    KL is summed over the latent dimension then averaged over the batch, which
    keeps its scale independent of ``latent_dim`` and makes ``beta`` comparable
    across runs with different latent sizes.

    Logvar is clamped to ``[-5, 5]`` before exponentiation to prevent numerical
    overflow: ``exp(5) ~ 148`` is ample dynamic range for a posterior std while
    leaving no headroom for the loss-explosion spikes that ``exp(15) ~ 3.3e6``
    allowed.

    Free bits
    ---------
    When ``free_bits > 0``, each latent dimension is allowed that many nats of KL
    "for free": the per-dimension (batch-averaged) KL is floored at ``free_bits``
    *inside the loss only*, so once a dimension drops below the floor its KL term
    contributes no gradient and the optimiser is no longer rewarded for collapsing
    it onto the prior.  This guards against posterior collapse (KL -> 0,
    reconstruction stuck at the predict-the-mean baseline), which becomes likely
    at large batch sizes where gradient noise no longer breaks the symmetry.

    The **returned** ``kl`` is always the true, unclamped KL so that logged values
    stay comparable across runs regardless of ``free_bits``.
    """
    import torch.nn.functional as F

    recon = F.mse_loss(xhat, x, reduction="mean")
    logvar = logvar.clamp(-5, 5)
    # per-dimension KL, averaged over the batch -> shape (latent_dim,)
    kl_per_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0)
    kl_true = kl_per_dim.sum()  # reported (sum over dim, mean over batch)

    if free_bits > 0.0:
        kl_for_loss = kl_per_dim.clamp(min=free_bits).sum()
    else:
        kl_for_loss = kl_true

    return recon + beta * kl_for_loss, recon, kl_true


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


from dataclasses import dataclass  # noqa: E402 (after TYPE_CHECKING guard imports above)


@dataclass
class VAEArtifact:
    """Trained VAE and the normalisation statistics needed to project new data.

    Returned by :func:`fit_vae` and consumed by :func:`transform_vae_latents`,
    :func:`save_vae`, and :func:`load_vae`.

    Attributes
    ----------
    model
        Trained ``nn.Module`` (encoder + decoder).  Call ``model.eval()``
        before any inference — :func:`transform_vae_latents` does this for you.
    mean
        Per-feature mean used for standardisation during training, shape
        ``(n_features,)``.  Must be applied to new data before encoding.
    std
        Per-feature std used for standardisation during training, shape
        ``(n_features,)``.  Features with near-zero variance were set to 1.0.
    device
        Device string (``"cpu"`` or ``"cuda"``) the model lives on.
    latent_dim
        Bottleneck size, stored for reference.
    """

    model: object          # nn.Module; typed as object to avoid top-level torch import
    mean: np.ndarray
    std: np.ndarray
    device: str
    latent_dim: int


def _train_vae(
    model,
    train_loader,
    optimizer,
    scaler,
    device: str,
    *,
    epochs: int,
    beta: float,
    free_bits: float,
    amp_enabled: bool,
    patience: int,
    min_delta: float,
    max_grad_norm: float,
    verbose: bool,
) -> None:
    """Run the VAE training loop, in place, with free-bits KL and early stopping.

    Shared by :func:`fit_vae` and :func:`fit_vae_latents` so the two entry points
    cannot drift out of sync.

    Gradient clipping
    -----------------
    When ``max_grad_norm > 0``, the global gradient norm is clipped before each
    optimiser step, so a single pathological batch cannot launch the weights into
    an overflow/``nan`` cascade.  Set to 0 to disable.

    Best-weight restoration
    -----------------------
    The model state at the lowest reconstruction loss seen is snapshotted and
    restored before returning.  This means a late loss spike (even one the run
    never recovers from) cannot silently leave you with degraded weights: the
    embedding is always taken from the best epoch, not the last one.

    Early stopping
    --------------
    Training halts when the mean reconstruction loss fails to improve by more than
    ``min_delta`` (absolute) for ``patience`` consecutive epochs.  This makes the
    epoch count effectively self-tuning: as the dataset grows there are more
    gradient steps per epoch, so convergence happens in fewer epochs and the loop
    stops on its own rather than burning time past the plateau.  Set
    ``patience=0`` to disable and always run the full ``epochs``.

    See :func:`_vae_loss` for the free-bits mechanism (``free_bits``).
    """
    import copy  # noqa: PLC0415

    import torch  # noqa: PLC0415

    best_recon = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0

    model.train()
    for epoch in range(1, epochs + 1):
        tot_loss = tot_recon = tot_kl = 0.0
        n_batches = 0

        for xb in train_loader:
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device, enabled=amp_enabled):
                xhat, mu, logvar = model(xb)
                loss, recon, kl = _vae_loss(xb, xhat, mu, logvar, beta, free_bits=free_bits)

            # Skip a pathological batch rather than letting nan/inf poison the weights
            if not torch.isfinite(loss):
                if verbose:
                    print(f"  warning: non-finite loss in epoch {epoch}, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)  # no-op when scaler disabled (CPU); required on CUDA
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            tot_loss += loss.item()
            tot_recon += recon.item()
            tot_kl += kl.item()
            n_batches += 1

        if n_batches == 0:
            raise FloatingPointError(
                f"all batches in epoch {epoch} produced non-finite loss; "
                "lower lr, lower max_grad_norm, or check input scaling"
            )

        mean_loss = tot_loss / n_batches
        mean_recon = tot_recon / n_batches
        mean_kl = tot_kl / n_batches

        if verbose:
            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"loss={mean_loss:.4f}  "
                f"recon={mean_recon:.4f}  "
                f"kl={mean_kl:.4f}"
            )

        # Early stopping on reconstruction loss + snapshot best weights
        if mean_recon < best_recon - min_delta:
            best_recon = mean_recon
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                if verbose:
                    print(
                        f"  early stopping at epoch {epoch}/{epochs} "
                        f"(recon improved < {min_delta} for {patience} epochs; "
                        f"best recon={best_recon:.4f})"
                    )
                break

    # Restore the best-seen weights so a late spike can't leave us on worse ones
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"  restored best weights (recon={best_recon:.4f})")


def fit_vae(
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
    free_bits: float = 0.5,
    patience: int = 5,
    min_delta: float = 1e-3,
    max_grad_norm: float = 5.0,
    device: Literal["cpu", "cuda", "auto"] = "auto",
    verbose: bool = True,
) -> tuple[np.ndarray, VAEArtifact]:
    """Train a VAE and return both the latent embedding and the trained model.

    This is the full-featured counterpart to :func:`fit_vae_latents`.  Use this
    when you need to project new / held-out data into the same latent space after
    training (e.g. train on mouse seqlets, project human seqlets).  The returned
    :class:`VAEArtifact` can be saved to disk with :func:`save_vae` and reloaded
    later with :func:`load_vae`.

    Parameters
    ----------
    X
        Training matrix of shape ``(n_seqlets, n_features)``.
    latent_dim, hidden, n_layers, beta, lr, epochs, batch_size, num_workers,
    use_amp, dropout, n_sample_stats, seed, device, verbose
        See :func:`fit_vae_latents` for full documentation of each parameter.

    Returns
    -------
    Z
        Float32 array of shape ``(n_seqlets, latent_dim)`` — posterior means
        for the training data.
    artifact
        :class:`VAEArtifact` bundling the trained model and normalisation
        statistics.  Pass to :func:`transform_vae_latents` to embed new data.

    Examples
    --------
    >>> Z_mouse, vae = fit_vae(mm_adata.X, latent_dim=10, epochs=50)
    >>> mm_adata.obsm["X_vae"] = Z_mouse
    >>>
    >>> Z_human = transform_vae_latents(hs_adata.X, vae)
    >>> hs_adata.obsm["X_vae"] = Z_human
    >>>
    >>> save_vae(vae, "mouse_vae.pt")
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

    if device == "auto":
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = device

    torch.manual_seed(seed)
    print(f"Training VAE on {_device} (latent_dim={latent_dim}, epochs={epochs}, beta={beta})...")

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

    in_dim = X.shape[1]
    model = _build_vae(in_dim, latent_dim, hidden, n_layers, dropout).to(_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    _amp_enabled = use_amp and _device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled)

    _train_vae(
        model,
        train_loader,
        optimizer,
        scaler,
        _device,
        epochs=epochs,
        beta=beta,
        free_bits=free_bits,
        amp_enabled=_amp_enabled,
        patience=patience,
        min_delta=min_delta,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
    )

    # Embed training data
    Z = _encode(model, dataset, _device, num_workers, _pin)
    print(f"VAE embedding complete. Shape: {Z.shape}")

    artifact = VAEArtifact(model=model, mean=mean, std=std, device=_device, latent_dim=latent_dim)
    return Z, artifact


def transform_vae_latents(
    X: np.ndarray | sp.spmatrix,
    artifact: VAEArtifact,
    *,
    batch_size: int = 8192,
    num_workers: int = 0,
) -> np.ndarray:
    """Project new seqlets into an existing VAE latent space.

    Applies the same per-feature standardisation used during training, then runs
    the encoder forward pass to obtain posterior means.  The decoder is not used.

    Parameters
    ----------
    X
        New data matrix of shape ``(n_seqlets, n_features)``.  Must have the
        same number of features as the training data.
    artifact
        :class:`VAEArtifact` returned by :func:`fit_vae` (or loaded with
        :func:`load_vae`).
    batch_size
        Inference batch size.  Larger values are faster; reduce if OOM.
    num_workers
        DataLoader worker processes.

    Returns
    -------
    Z
        Float32 array of shape ``(n_seqlets, latent_dim)``.

    Raises
    ------
    ValueError
        If ``X`` has a different number of features than the training data.

    Examples
    --------
    >>> Z_human = transform_vae_latents(hs_adata.X, vae)
    >>> hs_adata.obsm["X_vae"] = Z_human
    """
    n_features_train = artifact.mean.shape[0]
    if X.shape[1] != n_features_train:
        raise ValueError(
            f"Feature dimension mismatch: model was trained on {n_features_train} features, "
            f"but X has {X.shape[1]}."
        )

    dataset = _make_dataset(X, artifact.mean, artifact.std)
    _pin = artifact.device == "cuda"
    artifact.model.eval()
    Z = _encode(artifact.model, dataset, artifact.device, num_workers, _pin, batch_size=batch_size)
    print(f"VAE projection complete. Shape: {Z.shape}")
    return Z


def save_vae(artifact: VAEArtifact, path: str) -> None:
    """Save a :class:`VAEArtifact` to disk.

    Saves the model weights alongside the normalisation statistics so the full
    artifact can be reconstructed with :func:`load_vae` without retraining.

    Parameters
    ----------
    artifact
        Artifact returned by :func:`fit_vae`.
    path
        Output file path.  Conventionally ends in ``.pt``.

    Examples
    --------
    >>> save_vae(vae, "mouse_vae.pt")
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch is required to save a VAEArtifact.") from e

    torch.save(
        {
            "state_dict": artifact.model.state_dict(),
            "mean": artifact.mean,
            "std": artifact.std,
            "device": artifact.device,
            "latent_dim": artifact.latent_dim,
        },
        path,
    )
    print(f"VAE artifact saved to {path}")


def load_vae(
    path: str,
    in_dim: int,
    *,
    hidden: int = 512,
    n_layers: int = 2,
    dropout: float = 0.1,
    device: Literal["cpu", "cuda", "auto"] = "auto",
) -> VAEArtifact:
    """Load a :class:`VAEArtifact` from disk.

    The model architecture parameters (``hidden``, ``n_layers``, ``dropout``)
    must match those used during training.

    Parameters
    ----------
    path
        Path to a ``.pt`` file saved by :func:`save_vae`.
    in_dim
        Input feature dimension (number of motifs), i.e. ``adata.X.shape[1]``.
    hidden
        Must match the value used when training.
    n_layers
        Must match the value used when training.
    dropout
        Must match the value used when training.
    device
        Device to load the model onto.  ``"auto"`` uses CUDA if available.

    Returns
    -------
    VAEArtifact
        Ready to pass to :func:`transform_vae_latents`.

    Examples
    --------
    >>> vae = load_vae("mouse_vae.pt", in_dim=17995)
    >>> Z_human = transform_vae_latents(hs_adata.X, vae)
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch is required to load a VAEArtifact.") from e

    if device == "auto":
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = device

    checkpoint = torch.load(path, map_location=_device)
    latent_dim = checkpoint["latent_dim"]

    model = _build_vae(in_dim, latent_dim, hidden, n_layers, dropout).to(_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print(f"VAE artifact loaded from {path} (latent_dim={latent_dim}, device={_device})")
    return VAEArtifact(
        model=model,
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        device=_device,
        latent_dim=latent_dim,
    )


# ---------------------------------------------------------------------------
# Internal embedding helper (shared by fit_vae and transform_vae_latents)
# ---------------------------------------------------------------------------


def _encode(model, dataset, device: str, num_workers: int, pin_memory: bool, batch_size: int = 8192) -> np.ndarray:
    """Run encoder over a dataset and return stacked posterior means."""
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            mu = model.mu_head(model.enc(xb))
            parts.append(mu.cpu().numpy())
    return np.vstack(parts).astype(np.float32)


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
    free_bits: float = 0.5,
    patience: int = 5,
    min_delta: float = 1e-3,
    max_grad_norm: float = 5.0,
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
    free_bits
        Minimum KL (in nats) reserved per latent dimension, applied inside the
        loss only.  Once a dimension's batch-averaged KL falls below this floor,
        its KL gradient vanishes, so the optimiser is no longer rewarded for
        collapsing that dimension onto the prior.  This is the main guard against
        posterior collapse (KL -> 0, reconstruction stuck at the predict-the-mean
        baseline of 1.0 on standardised data), which is most likely at large
        batch sizes.  The default 0.5 sits well below the per-dimension KL of a
        healthy run (~0.7 nats/dim for latent_dim=10), so it does not interfere
        with normal training.  Set to 0.0 to recover the plain beta-VAE objective.
    patience
        Early-stopping patience: training halts if the mean reconstruction loss
        does not improve by more than ``min_delta`` for this many consecutive
        epochs.  Makes the epoch count self-tuning as the dataset grows.  Set to
        0 to disable and always run the full ``epochs``.
    min_delta
        Minimum absolute improvement in reconstruction loss that counts as
        progress for early stopping.
    max_grad_norm
        Global gradient-norm clip applied before each optimiser step.  Guards
        against loss-explosion spikes where one bad batch sends the weights to
        ``inf``/``nan``.  The default 5.0 is a safe value; set to 0 to disable.
        Note: the best-recon weights are snapshotted and restored at the end of
        training, so even a spike the run cannot recover from will not degrade
        the returned embedding.
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
    _train_vae(
        model,
        train_loader,
        optimizer,
        scaler,
        _device,
        epochs=epochs,
        beta=beta,
        free_bits=free_bits,
        amp_enabled=_amp_enabled,
        patience=patience,
        min_delta=min_delta,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
    )

    # ----------------------------------------------------------- embed (mu)
    Z = _encode(model, dataset, _device, num_workers, _pin)
    print(f"VAE embedding complete. Shape: {Z.shape}")
    return Z