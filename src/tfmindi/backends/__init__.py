"""Backend detection and configuration for CPU/GPU acceleration."""

from __future__ import annotations

import os
import warnings
from typing import Any, Literal

Backend = Literal["cpu", "gpu"]

# Global backend state
_backend: Backend | None = None
_gpu_available: bool | None = None


def _check_gpu_availability() -> bool:
    """Check if GPU acceleration packages are available."""
    global _gpu_available
    if _gpu_available is not None:
        return _gpu_available

    try:
        import cupy as cp  # type: ignore

        device_count = cp.cuda.runtime.getDeviceCount()
        _gpu_available = device_count > 0
        return _gpu_available  # type: ignore
    except ImportError:
        _gpu_available = False
        return False


def get_backend() -> str:
    """
    Get the current computational backend.

    Returns the backend based on the following priority:
    1. Explicitly set backend via set_backend()
    2. Environment variable TFMINDI_BACKEND
    3. Automatic detection based on GPU availability

    Returns
    -------
    Backend type: "cpu" or "gpu"
    """
    global _backend

    if _backend is not None:
        return _backend

    # Check environment variable
    env_backend = os.getenv("TFMINDI_BACKEND", "").lower()
    if env_backend in ["cpu", "gpu"]:
        _backend = env_backend  # type: ignore
        if _backend == "gpu" and not _check_gpu_availability():
            warnings.warn(
                "GPU backend requested but GPU packages not available. "
                "Install with 'pip install tfmindi[gpu]'. Falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )
            _backend = "cpu"
        return _backend  # type: ignore

    # Auto-detect based on availability
    _backend = "gpu" if _check_gpu_availability() else "cpu"
    return _backend


def set_backend(backend: Backend) -> None:
    """
    Explicitly set the computational backend.

    Parameters
    ----------
    backend
        Backend type: "cpu" or "gpu"

    Raises
    ------
    ValueError
        If backend is not supported
    ImportError
        If GPU backend is requested but packages are not available
    """
    global _backend

    if backend not in ["cpu", "gpu"]:
        raise ValueError(f"Invalid backend: {backend}. Must be 'cpu' or 'gpu'.")

    if backend == "gpu" and not _check_gpu_availability():
        raise ImportError(
            "GPU backend requested but required packages not available. Install with 'pip install tfmindi[gpu]'."
        )

    _backend = backend


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _check_gpu_availability()


__all__ = [
    "Backend",
    "get_backend",
    "set_backend",
    "is_gpu_available",
]
