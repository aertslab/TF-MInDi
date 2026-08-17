"""Backend detection and configuration for CPU/GPU acceleration."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any, Literal, TypeVar

Backend = Literal["cpu", "gpu"]

T = TypeVar("T")

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


def using_gpu() -> bool:
    """
    Check whether the GPU backend is both selected and usable.

    Resolved at call time rather than at import time, so a backend chosen after
    ``import tfmindi`` still takes effect.

    Returns
    -------
    True when the active backend is ``"gpu"`` and the GPU packages are importable.
    """
    return get_backend() == "gpu" and is_gpu_available()


def rapids_singlecell() -> Any:
    """
    Import :mod:`rapids_singlecell` at call time.

    Kept out of module scope so importing tfmindi never requires the GPU extra, and so
    an ImportError surfaces inside the ``try`` of :func:`run_accelerated`.

    Returns
    -------
    The :mod:`rapids_singlecell` module.
    """
    import rapids_singlecell as rsc  # type: ignore

    return rsc


def to_numpy(x: Any) -> Any:
    """
    Bring an array back to host memory.

    GPU libraries differ in whether they hand back a cupy array or a numpy one, so results
    crossing back into the CPU pipeline go through here rather than each call site
    re-testing for it.

    Parameters
    ----------
    x
        A cupy array, a numpy array, or anything else.

    Returns
    -------
    ``x.get()`` when ``x`` is device-resident, otherwise ``x`` unchanged.
    """
    return x.get() if hasattr(x, "get") else x


def run_accelerated(step: str, gpu_fn: Callable[[], T], cpu_fn: Callable[[], T]) -> T:
    """
    Run a step on the GPU when the GPU backend is active, otherwise on the CPU.

    Centralises the package convention for accelerated steps: the backend is resolved at
    call time, and *any* failure inside the GPU path warns and re-runs the step on the
    CPU. A missing driver, an unsupported argument or an out-of-memory error therefore
    degrades to the CPU result instead of aborting the pipeline.

    Parameters
    ----------
    step
        Human-readable name of the step, used in the fallback warning.
    gpu_fn
        Zero-argument callable running the GPU implementation.
    cpu_fn
        Zero-argument callable running the CPU implementation.

    Returns
    -------
    Whatever the chosen implementation returns.
    """
    if using_gpu():
        try:
            return gpu_fn()
        except Exception as e:  # noqa: BLE001 - any GPU failure must fall back, not propagate
            warnings.warn(f"GPU {step} failed: {e}. Falling back to CPU.", UserWarning, stacklevel=2)
    return cpu_fn()


__all__ = [
    "Backend",
    "get_backend",
    "set_backend",
    "is_gpu_available",
    "using_gpu",
    "rapids_singlecell",
    "to_numpy",
    "run_accelerated",
]
