"""Seqlet extraction and motif similarity preprocessing functions for TF-MInDi."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, cast, get_args

import numba
import numpy as np
import pandas as pd
from anndata import AnnData
from memelite import tomtom
from scipy import sparse
from tqdm import tqdm

from tfmindi.backends import is_gpu_available

EPS = 1e-8

# Seqlet-calling methods selectable via ``extract_seqlets(..., method=...)``. Kept as
# a Literal alias (for static ``method=`` typing) plus a runtime tuple derived from it
# via ``get_args`` so the list of valid names is written exactly once.
SeqletCallerMethod = Literal[
    "recursive_q99_abs_smooth",
    "recursive_raw",
    "hysteresis",
    "local_contrast",
    "wavelet_otsu",
    "finemo_fit_contrib",
]
SEQLET_CALLERS: tuple[SeqletCallerMethod, ...] = get_args(SeqletCallerMethod)


class _1DSeqletCaller(ABC):
    """Caller signature for methods that threshold the projected 1D attribution track."""

    @abstractmethod
    def __call__(self, X: np.ndarray) -> pd.DataFrame: ...


class _FitContribSeqletCaller(ABC):
    """Caller signature for methods that fit directly against the raw per-base contribution/one-hot arrays and a motif collection (e.g. fitting motif CWMs)."""

    @abstractmethod
    def __call__(self, contrib: np.ndarray, oh: np.ndarray, motifs: dict[str, np.ndarray]) -> pd.DataFrame: ...


# -----------------------------------------------------------------------------
# Signal / interval utilities (ported from the tfmindi-evaluations benchmark)
# -----------------------------------------------------------------------------


def _ensure_2d(X: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Project an input array to float64 shape (n_examples, length)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("Expected a 1D or 2D array with shape (n_examples, length).")
    return X


def _triangular_smooth(x: np.ndarray, window: int | None = 9) -> np.ndarray:
    """Smooth a 1D vector with a symmetric triangular kernel."""
    x = np.asarray(x, dtype=np.float64)
    if window is None or window <= 1 or x.size <= 1:
        return x.copy()
    if window % 2 == 0:
        raise ValueError("triangular_smooth requires an odd window size.")
    if window > x.size:
        window = int(x.size if x.size % 2 == 1 else x.size - 1)
        if window <= 1:
            return x.copy()
    half = window // 2
    kernel = np.arange(1, half + 2, dtype=np.float64)
    kernel = np.concatenate([kernel, kernel[-2::-1]])
    kernel = kernel / kernel.sum()
    return np.convolve(x, kernel, mode="same")


def _sliding_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Return the sliding window sum of x with the given window size."""
    x = np.asarray(x, dtype=np.float64)
    if window < 1 or window > x.shape[0]:
        return np.empty(0, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return cs[window:] - cs[:-window]


def _reciprocal_overlap(a0: int, a1: int, b0: int, b1: int) -> float:
    """Overlap fraction relative to the shorter of the two half-open intervals."""
    ov = max(0, min(a1, b1) - max(a0, b0))
    if ov <= 0:
        return 0.0
    return ov / max(min(a1 - a0, b1 - b0), 1)


def _robust_zscore(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Median/MAD z-score with std/max fallback."""
    x = np.asarray(x, dtype=np.float64)
    center = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - center)))
    if mad <= EPS:
        mad = float(np.std(x))
    if mad <= EPS:
        mad = max(float(np.max(np.abs(x))), 1.0)
    return (x - center) / mad, center, mad


def _standard_zscore(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Mean/std z-score with numerical floor."""
    x = np.asarray(x, dtype=np.float64)
    center = float(np.mean(x))
    scale = float(np.std(x))
    if scale <= EPS:
        scale = 1.0
    return (x - center) / scale, center, scale


def _normalize_tracks(X: np.ndarray, mode: str = "q99") -> np.ndarray:
    """Per-example normalization for 1D tracks."""
    X = _ensure_2d(X)
    if mode == "none":
        return X.copy()
    if mode == "q99":
        denom = np.quantile(np.abs(X), 0.99, axis=1, keepdims=True)
    elif mode == "maxabs":
        denom = np.max(np.abs(X), axis=1, keepdims=True)
    elif mode == "l2":
        denom = np.linalg.norm(X, axis=1, keepdims=True)
    else:
        raise ValueError("mode must be one of {'none', 'q99', 'maxabs', 'l2'}")
    denom = np.where(denom > EPS, denom, 1.0)
    return X / denom


# Shared caller output contract: one row per seqlet with these columns.
_SEQLET_COLS = ["example_idx", "start", "end", "attribution", "score"]


def _seqlet_df(rows) -> pd.DataFrame:
    """Build a sorted seqlet DataFrame from a list of row tuples."""
    out = pd.DataFrame(rows, columns=_SEQLET_COLS)
    if len(out) == 0:
        return pd.DataFrame(columns=_SEQLET_COLS)
    out[["example_idx", "start", "end"]] = out[["example_idx", "start", "end"]].astype(int)
    return out.sort_values(
        ["example_idx", "start", "end", "score"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def _dedup_seqlets(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (example_idx, start, end) intervals, keeping the highest score."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_SEQLET_COLS)
    out = df.copy()
    if "score" not in out:
        out["score"] = 1.0
    if "attribution" not in out:
        out["attribution"] = out["end"] - out["start"]
    out = out.sort_values("score", ascending=False).drop_duplicates(["example_idx", "start", "end"], keep="first")
    return out.sort_values(["example_idx", "start", "end"]).reset_index(drop=True)


def _merge_close(intervals, merge_gap: int = 2, max_len: int = 40) -> list:
    """Merge sorted intervals whose gaps are within merge_gap, capping length at max_len."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda t: (t[0], t[1]))
    out = [[intervals[0][0], intervals[0][1], intervals[0][2]]]
    for s, e, score in intervals[1:]:
        last = out[-1]
        if s - last[1] <= merge_gap and e - last[0] <= max_len:
            last[1] = max(last[1], e)
            last[2] = max(last[2], score)
        else:
            out.append([s, e, score])
    return [(int(s), int(e), float(score)) for s, e, score in out]


# -----------------------------------------------------------------------------
# Seqlet callers (ported from the tfmindi-evaluations benchmark)
#
# Every caller takes a 2D array ``X`` of shape (n_examples, length) — the 1D
# projected attribution track — and returns a DataFrame with columns
# ``["example_idx", "start", "end", "attribution", "score"]``.
# -----------------------------------------------------------------------------


class rec_q99_smooth_abs(_1DSeqletCaller):  # noqa: D101
    def __call__(
        self,
        X: np.ndarray,
        smooth_window: int = 9,
        threshold: float = 0.05,
        min_seqlet_len: int = 4,
        max_seqlet_len: int = 25,
        additional_flanks: int = 3,
    ) -> pd.DataFrame:
        """Core backbone: triangular smooth -> q99 normalise -> recursive abs caller."""
        X = _ensure_2d(X)
        Xs = np.array([_triangular_smooth(x, smooth_window) for x in X], dtype=np.float64)
        Xn = _normalize_tracks(Xs, mode="q99")

        raw = recursive_seqlets(
            np.abs(Xn),
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
        )
        if len(raw) == 0:
            return _seqlet_df([])

        raw = raw.rename(columns={"p-value": "score"})
        raw["score"] = -np.log10(np.clip(raw["score"].astype(float), 1e-300, None))
        return _dedup_seqlets(raw)


class recursive_raw(_1DSeqletCaller):  # noqa: D101
    def __call__(
        self,
        X: np.ndarray,
        threshold: float = 0.05,
        min_seqlet_len: int = 4,
        max_seqlet_len: int = 25,
        additional_flanks: int = 3,
    ) -> pd.DataFrame:
        """Baseline caller: recursive seqlets on the raw *signed* 1D track.

        Reproduces the original TF-MINDI ``extract_seqlets`` behaviour — the in-tree
        tangermeme-style :func:`recursive_seqlets` on the signed ``(contrib*oh).sum(1)``
        track. The p-value is converted to ``score = -log10(p)`` to match the caller
        contract.
        """
        X = _ensure_2d(X)
        raw = recursive_seqlets(
            X,
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
        )
        if raw is None or len(raw) == 0:
            return _seqlet_df([])
        raw = raw.rename(columns={"p-value": "score"})
        raw["score"] = -np.log10(np.clip(raw["score"].astype(float), 1e-300, None))
        return _dedup_seqlets(raw)


class hysteresis(_1DSeqletCaller):  # noqa: D101
    def __call__(
        self,
        X: np.ndarray,
        smooth_window: int = 9,
        seed_z: float = 2.5,
        grow_z: float = 1.0,
        min_seqlet_len: int = 4,
        max_seqlet_len: int = 25,
        merge_gap: int = 2,
    ) -> pd.DataFrame:
        """Two-threshold local caller: high-z seed, lower-z growth on smoothed abs(track)."""
        X = _ensure_2d(X)
        rows = []
        for ex, raw in enumerate(X):
            abs_track = np.abs(_triangular_smooth(raw, window=smooth_window))
            z, _, _ = _standard_zscore(abs_track)
            high = z >= seed_z
            low = z >= grow_z
            pos = 0
            intervals = []
            while pos < len(raw):
                if not low[pos]:
                    pos += 1
                    continue
                start = pos
                has_seed = bool(high[pos])
                pos += 1
                while pos < len(raw) and low[pos]:
                    has_seed = has_seed or bool(high[pos])
                    pos += 1
                end = pos
                if not has_seed or end - start < min_seqlet_len:
                    continue
                if end - start > max_seqlet_len:
                    sums = _sliding_sum(abs_track[start:end], max_seqlet_len)
                    best = int(np.argmax(sums)) + start
                    start, end = best, best + max_seqlet_len
                intervals.append((start, end, float(np.max(z[start:end]))))
            intervals = _merge_close(intervals, merge_gap=merge_gap, max_len=max_seqlet_len)
            for s, e, score in intervals:
                rows.append((ex, s, e, float(np.sum(raw[s:e])), score))
        return _seqlet_df(rows)


class local_contrast(_1DSeqletCaller):  # noqa: D101
    def __call__(
        self,
        X: np.ndarray,
        windows: Sequence[int] = (10, 16, 24),
        smooth_window: int = 9,
        seed_z: float = 4.0,
        expand_z: float = 1.25,
        min_seqlet_len: int = 4,
        max_seqlet_len: int = 25,
        merge_gap: int = 2,
        nms_reciprocal_overlap: float = 0.50,
    ) -> pd.DataFrame:
        """Sliding-window local contrast caller.

        For each window w in ``windows``, score = (mean(core) - mean(flanks)) * sqrt(w).
        Local maxima above seed_z are taken as candidates, refined by expanding into
        neighbouring positions with absolute z-score >= expand_z, then deduplicated by
        reciprocal overlap.
        """
        X = _ensure_2d(X)
        rows = []
        for ex, raw in enumerate(X):
            base = _triangular_smooth(raw, window=smooth_window)
            abs_track = np.abs(base)
            L = len(abs_track)
            z_abs, _, _ = _robust_zscore(abs_track)

            # Prefix sum for fast window mean computation.
            cs = np.concatenate([[0.0], np.cumsum(abs_track)])

            candidates = []
            for w in tuple(int(w) for w in windows):
                if w > L:
                    continue
                n_pos = L - w + 1
                s_idx = np.arange(n_pos)
                e_idx = s_idx + w

                core = (cs[e_idx] - cs[s_idx]) / w

                l0 = np.maximum(s_idx - w, 0)
                l_len = np.maximum(s_idx - l0, 1)
                left = (cs[s_idx] - cs[l0]) / l_len

                r1 = np.minimum(e_idx + w, L)
                r_len = np.maximum(r1 - e_idx, 1)
                right = (cs[r1] - cs[e_idx]) / r_len

                scores = (core - 0.5 * (left + right)) * np.sqrt(w)
                z, _, _ = _robust_zscore(scores)

                for i in range(z.shape[0]):
                    left_ok = i == 0 or z[i] >= z[i - 1]
                    right_ok = i == z.shape[0] - 1 or z[i] > z[i + 1]
                    if z[i] >= seed_z and left_ok and right_ok:
                        candidates.append((float(z[i]), i, i + w))

            candidates.sort(key=lambda t: t[0], reverse=True)
            selected: list[tuple[int, int, float]] = []
            for score, s, e in candidates:
                # Expand into neighbouring high-attribution positions.
                while s > 0 and z_abs[s - 1] >= expand_z and e - s < max_seqlet_len:
                    s -= 1
                while e < L and z_abs[e] >= expand_z and e - s < max_seqlet_len:
                    e += 1
                if e - s < min_seqlet_len:
                    continue
                if any(_reciprocal_overlap(s, e, ps, pe) >= nms_reciprocal_overlap for ps, pe, _ in selected):
                    continue
                selected.append((s, e, score))
            selected = _merge_close(selected, merge_gap=merge_gap, max_len=max_seqlet_len)
            for s, e, score in selected:
                rows.append((ex, s, e, float(np.sum(raw[s:e])), score))
        return _seqlet_df(rows)


def _wo_mask_to_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert a boolean mask into a list of contiguous half-open intervals."""
    intervals = []
    in_region = False
    start = 0
    for j in range(len(mask)):
        if mask[j] and not in_region:
            start = j
            in_region = True
        elif not mask[j] and in_region:
            intervals.append((start, j))
            in_region = False
    if in_region:
        intervals.append((start, len(mask)))
    return intervals


def _wo_merge_intervals(intervals: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge intervals separated by gaps of at most ``max_gap`` positions."""
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(int(s), int(e)) for s, e in merged]


def _wo_wavelet_denoise(
    sig: np.ndarray,
    wavelet: str = "coif2",
    max_level: int = 4,
    threshold_mode: str = "soft",
    threshold_scale: float = 1.7,
) -> np.ndarray:
    """Universal-threshold wavelet denoising of a 1D signal (PyWavelets)."""
    import pywt

    wv = pywt.Wavelet(wavelet)  # type: ignore
    natural_level = pywt.dwt_max_level(len(sig), wv.dec_len)
    level = min(natural_level, max_level) if max_level else natural_level
    if level < 1:
        return sig.copy()
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    if sigma == 0:
        sigma = 1e-8
    thresh = threshold_scale * sigma * np.sqrt(2 * np.log(len(sig)))
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, value=thresh, mode=threshold_mode))
    reconstructed = pywt.waverec(denoised_coeffs, wavelet)
    return reconstructed[: len(sig)]


def _wo_otsu_threshold(sig: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's method: threshold that maximises between-class variance."""
    sig_min, sig_max = sig.min(), sig.max()
    if sig_max - sig_min < 1e-10:
        return float(sig_max)
    hist, bin_edges = np.histogram(sig, bins=n_bins, range=(sig_min, sig_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return float(sig_max)
    best_thresh = sig_max
    best_variance = -1.0
    weight_bg = 0.0
    sum_bg = 0.0
    sum_total = np.sum(hist * bin_centers)
    for i in range(n_bins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            best_thresh = bin_centers[i]
    return float(best_thresh)


def _wo_refine_boundaries_raw(
    intervals: list[tuple[int, int]],
    raw: np.ndarray,
    expand: int = 2,
    contract_frac: float = 0.0,
) -> list[tuple[int, int]]:
    """Refine boundaries against the raw ``abs`` signal (expand, then contract weak edges)."""
    result = []
    for s, e in intervals:
        peak_raw = raw[s:e].max()
        if peak_raw <= 0:
            result.append((s, e))
            continue
        cutoff = peak_raw * contract_frac
        new_s, new_e = s, e
        for _ in range(expand):
            if new_s > 0 and raw[new_s - 1] > cutoff:
                new_s -= 1
            else:
                break
        for _ in range(expand):
            if new_e < len(raw) and raw[new_e] > cutoff:
                new_e += 1
            else:
                break
        while new_s < new_e - 4 and raw[new_s] < cutoff * 0.5:
            new_s += 1
        while new_e > new_s + 4 and raw[new_e - 1] < cutoff * 0.5:
            new_e -= 1
        if new_e - new_s >= 4:
            result.append((new_s, new_e))
        elif e - s >= 4:
            result.append((s, e))
    return result


class wavelet_otsu(_1DSeqletCaller):  # noqa: D101
    def __call__(
        self,
        X: np.ndarray,
        wavelet: str = "coif2",
        max_level: int = 4,
        threshold_mode: str = "soft",
        threshold_scale: float = 1.7,
        otsu_weight: float = 1.3,
        min_seqlet_len: int = 4,
        max_gap: int = 2,
        refine_expand: int = 2,
        refine_contract: float = 0.0,
    ) -> pd.DataFrame:
        """Wavelet-denoise + Otsu-threshold seqlet caller (PyWavelets).

        Per example: take ``abs`` of the signed track, wavelet-denoise, threshold with
        (weighted) Otsu, convert the mask to intervals, merge small gaps, refine
        boundaries against the raw ``abs`` signal, and drop intervals shorter than
        ``min_seqlet_len``. ``pywt`` is imported lazily inside the denoiser.
        """
        X = _ensure_2d(X)
        rows = []
        for ex, raw_signed in enumerate(X):
            raw = np.abs(raw_signed)

            denoised = _wo_wavelet_denoise(
                raw,
                wavelet=wavelet,
                max_level=max_level,
                threshold_mode=threshold_mode,
                threshold_scale=threshold_scale,
            )
            denoised = np.maximum(denoised, 0)

            thresh = _wo_otsu_threshold(denoised) * otsu_weight
            above = denoised > thresh
            intervals = _wo_mask_to_intervals(above)
            intervals = _wo_merge_intervals(intervals, max_gap)

            if refine_expand > 0 or refine_contract > 0:
                intervals = _wo_refine_boundaries_raw(
                    intervals, raw, expand=refine_expand, contract_frac=refine_contract
                )

            intervals = [(s, e) for s, e in intervals if (e - s) >= min_seqlet_len]
            for s, e in intervals:
                attr = float(np.sum(raw_signed[s:e]))
                rows.append((ex, int(s), int(e), attr, abs(attr)))
        return _seqlet_df(rows)


# -----------------------------------------------------------------------------
# FINEMO SEQLET CALLING
# -----------------------------------------------------------------------------


@dataclass
class _FinemoMotifMetadata:
    motif_name: str
    strand: str
    sign: int
    motif_start: int
    motif_end: int


def _prepare_motifs_for_finemo(
    motifs: dict[str, np.ndarray],
    ic_trim_threshold: float = 0.2,
    background: tuple[float, float, float, float] | None = None,
    pseudocount: float = 1e-3,
    include_rc: bool = True,
    include_neg: bool = True,
):
    """Convert real motif PFMs into hcwm-style, information-content-scaled matrices.

    Each PFM is rescaled per position by its information content relative to
    `background` (in bits), then the whole matrix is normalized to unit L2 norm,
    matching how TF-MoDISco hcwms are scaled. Since real motifs have different
    native widths (unlike TF-MoDISco patterns, which all share one seqlet window),
    every motif is zero-padded (flanked) to the width of the widest input motif so
    all motifs can be stacked into a single array; `trim_masks` marks which
    positions are real motif content (0 for both zero-padding and IC-trimmed flank).
    Trimming keeps the span between the leftmost and rightmost position whose
    information content passes `ic_trim_threshold`, rather than a fraction of a
    per-motif max, since IC has an absolute, comparable scale (0-2 bits for DNA)
    across motifs.

    Parameters
    ----------
    motifs
        Mapping of motif_id -> PFM array of shape (4, W) in A, C, G, T order.
        Each column (position) is assumed to sum to 1.
    ic_trim_threshold
        Positions with information content (in bits) below this value are
        considered flank and excluded from [motif_start, motif_end).
    background
        Background nucleotide frequencies, shape (4,). Defaults to uniform (0.25 each).
    pseudocount
        Added to each PFM column before renormalizing, to avoid log(0) for zero-count bases.
    include_rc
        Whether to also include the reverse complement of each motif.
    include_neg
        Whether to also include a sign-flipped (y-axis mirrored) copy of each
        motif, to match binding sites where the model assigns negative
        (repressive) importance to the same base pattern. Negation doesn't
        change trim coordinates or motif_scale, only the sign of the values.

    Returns
    -------
    motif_data: list[_FinemoMotifMetadata]
        Metadata with fields: motif_name, strand, sign, motif_start,
        motif_end. motif_start/motif_end are coordinates within the
        common, zero-padded width.
    icms : ndarray, shape (M, 4, W_max)
        Information-content-scaled matrices (unit L2 norm), zero-padded to the
        widest input motif.
    trim_masks : ndarray, shape (M, W_max)
        Binary masks (1 = real, IC-passing motif content, 0 = zero-padding or
        IC-trimmed flank).
    """
    if background is None:
        a_background = np.full(4, 0.25)
    else:
        if len(background) != 4:
            raise ValueError("Background need a length of 4, one value for each nucleotide.")
        a_background = np.array(background)
    a_background = a_background.reshape(4, 1)

    # compute IC-scaled matrix and trim motif based on IC
    max_width = 0
    ic_scaled_trimmed_motifs: dict[str, tuple[np.ndarray, int, int, int]] = {}
    for motif_id, pfm in motifs.items():
        pfm = pfm.astype(np.float64) + pseudocount
        pfm = pfm / pfm.sum(axis=0, keepdims=True)
        assert isinstance(pfm, np.ndarray)

        icm_raw = pfm * np.log2(pfm / a_background)
        icm_norm = np.sqrt((icm_raw**2).sum())
        icm_fwd = icm_raw / icm_norm
        assert isinstance(icm_fwd, np.ndarray)

        ic = icm_raw.sum(axis=0)
        motif_len = icm_fwd.shape[1]
        pass_inds = np.where(ic >= ic_trim_threshold)[0]
        if len(pass_inds) == 0:
            start_fwd, end_fwd = 0, motif_len
        else:
            start_fwd, end_fwd = int(pass_inds.min()), int(pass_inds.max()) + 1

        ic_scaled_trimmed_motifs[motif_id] = (icm_fwd, start_fwd, end_fwd, motif_len)
        max_width = max(max_width, motif_len)

    motif_data: list[_FinemoMotifMetadata] = []

    n_icms = len(ic_scaled_trimmed_motifs)
    if include_rc:
        n_icms *= 2
    if include_neg:
        n_icms *= 2
    icms = np.empty(
        (n_icms, 4, max_width),
        dtype=np.float16,
    )
    trim_masks = np.empty((n_icms, max_width), dtype=np.int8)

    i = 0
    for motif_name, (icm_fwd, start_fwd, end_fwd, motif_len) in ic_scaled_trimmed_motifs.items():
        pad_left = (max_width - motif_len) // 2
        pad_right = max_width - motif_len - pad_left

        icm_fwd_padded = np.pad(icm_fwd, ((0, 0), (pad_left, pad_right)))
        start_fwd_g, end_fwd_g = start_fwd + pad_left, end_fwd + pad_left

        trim_mask_fwd = np.zeros(max_width, dtype=np.int8)
        trim_mask_fwd[start_fwd_g:end_fwd_g] = 1

        strand_variants = [("+", icm_fwd_padded, trim_mask_fwd, start_fwd_g, end_fwd_g)]

        if include_rc:
            icm_rev_padded = icm_fwd_padded[::-1, ::-1]
            trim_mask_rev = trim_mask_fwd[::-1]
            start_rev_g, end_rev_g = max_width - end_fwd_g, max_width - start_fwd_g
            strand_variants.append(("-", icm_rev_padded, trim_mask_rev, start_rev_g, end_rev_g))

        for strand, icm_padded, trim_mask, start_g, end_g in strand_variants:
            sign_variants = [(1, icm_padded)]
            if include_neg:
                sign_variants.append((-1, -icm_padded))

            for sign, icm_signed in sign_variants:
                motif_data.append(
                    _FinemoMotifMetadata(
                        motif_name=motif_name,
                        strand=strand,
                        sign=sign,
                        motif_start=start_g,
                        motif_end=end_g,
                    )
                )

                icms[i] = icm_signed
                trim_masks[i] = trim_mask
                i += 1
    if i != icms.shape[0]:
        raise RuntimeError("Fewer motifs than expected in icms, this is a bug in the code!")

    return motif_data, icms, trim_masks


def _merge_overlapping_hits(
    hits_df: pd.DataFrame,
    group_col: str = "peak_id",
    start_col: str = "start",
    end_col: str = "end",
) -> pd.DataFrame:
    """Collapse overlapping hits within the same peak into a single hit.

    `start`/`end` become the union of the cluster; every other column is
    kept as a list of its per-hit values, in the same order across columns,
    so e.g. `motif_ids[i]` and `hit_coefficients[i]` describe the same hit.

    Parameters
    ----------
    hits_df
        hits pandas dataframe from finemo fit_contrib function.
    group_col
        peak idx column name
    start_col
        start column name
    end_col
        end column name

    Returns
    -------
    hits_df
        A new hits_df with collapsed hits
    """
    df = hits_df.sort_values([group_col, start_col]).reset_index(drop=True).copy()

    # running max of `end` seen so far within each peak, in start-sorted order
    running_max_end = df.groupby(group_col)[end_col].cummax()
    # a new cluster starts when the peak changes, or when this hit's start
    # is past the furthest end reached so far (i.e. there's a coordinate gap)
    is_new_cluster = (df[group_col] != df[group_col].shift()) | (df[start_col] > running_max_end.shift())
    cluster_id = is_new_cluster.cumsum()

    merged = df.groupby(cluster_id).agg(
        {
            group_col: "first",
            start_col: "min",  # take the left most boundry
            end_col: "max",  # take the right most boundry
            # other columns will be turned into a list
            # this preserves all metadata for overlapping hits
            **{k: list for k in df.columns if k not in [group_col, start_col, end_col]},
        }
    )
    return merged.reset_index(drop=True)


class finemo_fit_contrib(_FitContribSeqletCaller):  # noqa: D101
    def __call__(
        self,
        contrib: np.ndarray,
        oh: np.ndarray,
        motifs: dict[str, np.ndarray],
        # motif preparation params
        ic_trim_threshold: float = 0.2,
        ic_background: tuple[float, float, float, float] | None = None,
        pseudocount: float = 0.001,
        include_rc: bool = True,
        include_neg: bool = True,
        # finemo params
        compile_optimizer: bool = True,
        batch_size: int = 500,
    ) -> pd.DataFrame:
        """Call hits by fitting contribution scores using motifs as features using Finemo.

        Parameters
        ----------
        contrib
            Contribution scores array with shape (n_examples, 4, length)
        oh
            One-hot encoded sequences array with shape (n_examples, 4, length)
        motifs
            Mapping of motif_id -> PFM array of shape (4, W) in A, C, G, T order.
            Each column (position) is assumed to sum to 1.
        ic_trim_threshold
            Positions with information content (in bits) below this value are
            considered flank and excluded from [motif_start, motif_end).
        ic_background
            Background nucleotide frequencies, shape (4,). Defaults to uniform (0.25 each).
        pseudocount
            Added to each PFM column before renormalizing, to avoid log(0) for zero-count bases.
        include_rc
            Whether to also include the reverse complement of each motif.
        include_neg
            Whether to also include a sign-flipped (y-axis mirrored) copy of each
            motif, to match binding sites where the model assigns negative
            (repressive) importance to the same base pattern. Negation doesn't
            change trim coordinates or motif_scale, only the sign of the values.
        compile_optimizer
            Whether to compile the finemo optimizer or not.
        batch_size
            Number of regions to process simultaneously.
            Lower this number if you run out of memory.

        """
        if not is_gpu_available():
            print("WARNING: No GPU available! Finemo is GPU optimized.")

        import finemo

        print("Preparing motifs for finemo ...")
        motif_metadata, icms, trim_masks = _prepare_motifs_for_finemo(
            motifs=motifs,
            ic_trim_threshold=ic_trim_threshold,
            background=ic_background,
            pseudocount=pseudocount,
            include_rc=include_rc,
            include_neg=include_neg,
        )

        print("Fitting contribution scores ...")
        pl_hits_df, _ = finemo.hitcaller.fit_contribs(
            cwms=icms,
            contribs=contrib,
            sequences=oh,
            cwm_trim_mask=trim_masks,
            use_hypothetical=True,
            lambdas=np.repeat(0.7, icms.shape[0]),
            compile_optimizer=compile_optimizer,
            batch_size=batch_size,
        )
        pd_hits_df = pl_hits_df.to_pandas()

        # genomic start/end of each hit = hit_start (in-peak offset) + motif's own start/end (i.e., taking into account the masking)
        motif_starts = pd_hits_df["motif_id"].astype(int).map(lambda m: motif_metadata[cast(int, m)].motif_start)
        motif_ends = pd_hits_df["motif_id"].astype(int).map(lambda m: motif_metadata[cast(int, m)].motif_end)
        pd_hits_df["start"] = pd_hits_df["hit_start"] + motif_starts
        pd_hits_df["end"] = pd_hits_df["start"] + (motif_ends - motif_starts)

        pd_hits_df = _merge_overlapping_hits(pd_hits_df)

        idcs = pd_hits_df["peak_id"].to_numpy(dtype=int)
        starts = pd_hits_df["start"].to_numpy(dtype=int)
        ends = pd_hits_df["end"].to_numpy(dtype=int)

        # "motif_id"/"hit_coefficient" are list-valued post-merge (one entry per
        # overlapping hit collapsed into this row; see _merge_overlapping_hits).
        # Both are rendered as a comma-joined string rather than reduced to a scalar.
        track = (contrib * oh).sum(1)
        attribution = [float(track[ex, s:e].sum()) for ex, s, e in zip(idcs, starts, ends, strict=True)]

        seqlets_df = pd.DataFrame(
            data={
                "example_idx": idcs,
                "start": starts,
                "end": ends,
                "attribution": attribution,
                "score": pd_hits_df["hit_coefficient"].apply(lambda cs: ", ".join(f"{c:.4g}" for c in cs)),
                "finemo_hit_motif_names": pd_hits_df["motif_id"].apply(
                    lambda ids: ", ".join(motif_metadata[int(i)].motif_name[1] for i in ids)
                ),
            }
        )
        return seqlets_df


# Registry mapping method name -> caller instance. Each caller's ``__call__`` supplies
# its own parameter defaults, so any ``method_kwargs`` simply override those defaults.
# Whether a given entry is a `_1DSeqletCaller` or a `_FitContribSeqletCaller` is what
# `extract_seqlets` uses (via `isinstance`) to decide how to invoke it.
_SEQLET_CALLER_REGISTRY: dict[SeqletCallerMethod, _1DSeqletCaller | _FitContribSeqletCaller] = {
    "recursive_q99_abs_smooth": rec_q99_smooth_abs(),
    "recursive_raw": recursive_raw(),
    "hysteresis": hysteresis(),
    "local_contrast": local_contrast(),
    "wavelet_otsu": wavelet_otsu(),
    "finemo_fit_contrib": finemo_fit_contrib(),
}
assert tuple(_SEQLET_CALLER_REGISTRY) == SEQLET_CALLERS


def _lookup_seqlet_caller(method: SeqletCallerMethod) -> _1DSeqletCaller | _FitContribSeqletCaller:
    """Return the registered caller instance for ``method``.

    The result is either a `_1DSeqletCaller` (called with the projected 1D track) or
    a `_FitContribSeqletCaller` (called with the raw ``contrib``/``oh``/``motifs``) —
    callers should ``isinstance``-check the result to decide which. Raises
    ``ValueError`` for an unknown ``method``.
    """
    try:
        return _SEQLET_CALLER_REGISTRY[method]
    except KeyError:
        raise ValueError(f"method must be one of {SEQLET_CALLERS}; got {method!r}.") from None


def get_example_idx(adata: AnnData, seqlet_idx: int) -> int:
    """
    Get the index for an example associated with a seqlet.

    Parameters
    ----------
    adata
        AnnData object containing seqlet data with unique examples storage
    seqlet_idx
        Index of the seqlet (row index in adata.obs)

    Returns
    -------
    index integer value

    """
    if "unique_examples" not in adata.uns:
        raise ValueError("No unique_examples found in adata.uns. Use the new storage format.")
    if "example_oh_idx" not in adata.obs.columns:
        raise ValueError("No example_oh_idx found in adata.obs. Use the new storage format.")

    example_idx = int(adata.obs["example_oh_idx"].iloc[seqlet_idx])

    return example_idx


def get_example_oh(adata: AnnData, seqlet_idx: int) -> np.ndarray:
    """
    Get the one-hot sequence for an example associated with a seqlet.

    Parameters
    ----------
    adata
        AnnData object containing seqlet data with unique examples storage
    seqlet_idx
        Index of the seqlet (row index in adata.obs)

    Returns
    -------
    One-hot sequence array with shape (4, sequence_length)
    """
    if "unique_examples" not in adata.uns:
        raise ValueError("No unique_examples found in adata.uns. Use the new storage format.")
    if "example_oh_idx" not in adata.obs.columns:
        raise ValueError("No example_oh_idx found in adata.obs. Use the new storage format.")

    example_idx = get_example_idx(adata, seqlet_idx)
    return adata.uns["unique_examples"]["oh"][example_idx]


def get_example_contrib(adata: AnnData, seqlet_idx: int) -> np.ndarray:
    """
    Get the contribution scores for an example associated with a seqlet.

    Parameters
    ----------
    adata
        AnnData object containing seqlet data with unique examples storage
    seqlet_idx
        Index of the seqlet (row index in adata.obs)

    Returns
    -------
    Contribution scores array with shape (4, sequence_length)
    """
    if "unique_examples" not in adata.uns:
        raise ValueError("No unique_examples found in adata.uns. Use the new storage format.")
    if "example_contrib_idx" not in adata.obs.columns:
        raise ValueError("No example_contrib_idx found in adata.obs. Use the new storage format.")

    example_idx = get_example_idx(adata, seqlet_idx)
    return adata.uns["unique_examples"]["contrib"][example_idx]


def extract_seqlets(
    contrib: np.ndarray,
    oh: np.ndarray,
    method: SeqletCallerMethod = "recursive_q99_abs_smooth",
    motifs: dict[str, np.ndarray] | None = None,
    **method_kwargs,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """
    Extract, scale, and process seqlets from saliency maps.

    Seqlets are called from either the projected 1D attribution track
    ``(contrib * oh).sum(1)`` or, for ``"finemo_fit_contrib"``, directly from the raw
    ``contrib``/``oh``/``motifs``, using the selected ``method``; each seqlet's
    contribution matrix is then normalized by its maximum absolute contribution value
    and sign-corrected.

    Parameters
    ----------
    contrib
        Contribution scores array with shape (n_examples, 4, length)
    oh
        One-hot encoded sequences array with shape (n_examples, 4, length)
    method
        Seqlet-calling algorithm to use. One of:

        - ``"recursive_q99_abs_smooth"`` (default): triangular-smooth, per-example
          q99 normalisation, then the recursive caller on ``abs(track)``.
          Accepts ``smooth_window``, ``threshold`` (0.05), ``min_seqlet_len``,
          ``max_seqlet_len``, ``additional_flanks`` (3).
        - ``"recursive_raw"``: recursive caller on the raw signed track
          (reproduces the previous TF-MInDi default behaviour). Same knobs as above.
        - ``"hysteresis"``: two-threshold local caller. Accepts ``smooth_window``,
          ``seed_z`` (2.5), ``grow_z`` (1.0), ``min_seqlet_len``, ``max_seqlet_len``,
          ``merge_gap``.
        - ``"local_contrast"``: multi-scale sliding-window contrast caller. Accepts
          ``windows``, ``smooth_window``, ``seed_z`` (4.0), ``expand_z``, ...
        - ``"wavelet_otsu"``: wavelet-denoise + Otsu-threshold caller (needs
          PyWavelets). Accepts ``wavelet``, ``threshold_scale`` (1.7),
          ``otsu_weight``, ``min_seqlet_len``, ...
        - ``"finemo_fit_contrib"``: fits ``motifs`` (after IC scaling) directly against the raw
          ``contrib``/``oh`` arrays via the ``finemo`` package (GPU-optimized), instead
          of thresholding the projected 1D track. Requires ``motifs`` to be given.
          Accepts ``ic_trim_threshold`` (0.2), ``ic_background``, ``pseudocount``,
          ``include_rc``, ``include_neg``, ``compile_optimizer``, ``batch_size``.
    motifs
        Mapping of motif_id -> PFM array of shape (4, W) in A, C, G, T order, each
        column summing to 1. Only used (and required) when ``method="finemo_fit_contrib"``.
    **method_kwargs
        Method-specific hyperparameters forwarded to the selected caller, overriding
        its defaults, e.g. ``threshold=0.1`` for the recursive methods,
        ``seed_z=3.0`` for ``hysteresis``, ``threshold_scale=2.0`` for
        ``wavelet_otsu``. Passing a keyword the chosen caller does not accept raises
        a ``TypeError``. See the per-method caller classes for the full parameter
        lists (``rec_q99_smooth_abs``, ``recursive_raw``, ``hysteresis``,
        ``local_contrast``, ``wavelet_otsu``, ``finemo_fit_contrib``).

    Returns
    -------
    - DataFrame with seqlet coordinates and scores
      [example_idx, start, end, attribution, score]
    - List of processed seqlet contribution matrices

    Examples
    --------
    >>> seqlets_df, seqlet_matrices = extract_seqlets(contrib, oh)
    >>> print(seqlets_df.columns.tolist())
    ['example_idx', 'start', 'end', 'attribution', 'score']
    >>> # switch caller and tune it in one call
    >>> seqlets_df, seqlet_matrices = extract_seqlets(contrib, oh, method="hysteresis", seed_z=3.0)
    >>> # fit known motifs directly against the contribution scores via finemo
    >>> seqlets_df, seqlet_matrices = extract_seqlets(contrib, oh, method="finemo_fit_contrib", motifs=motifs)
    """
    assert contrib.shape == oh.shape, "Contribution and one-hot arrays must have the same shape"
    caller = _lookup_seqlet_caller(method)
    bound = partial(caller, **method_kwargs)
    if isinstance(caller, _FitContribSeqletCaller):
        if motifs is None:
            raise ValueError(f"method={method!r} requires `motifs` to be provided.")
        seqlets_df = bound(contrib, oh, motifs)
    else:
        seqlets_df = bound((contrib * oh).sum(1))

    # extract and normalize contribution scores
    seqlet_matrices: list[np.ndarray] = []

    for _, (ex_idx, start, end) in tqdm(
        seqlets_df[["example_idx", "start", "end"]].iterrows(),
        total=len(seqlets_df),
        desc="Processing seqlets",
    ):
        # Extract contribution scores and one-hot sequences for this seqlet
        X = contrib[ex_idx, :, start:end]  # (4, seqlet_length)
        O = oh[ex_idx, :, start:end]  # (4, seqlet_length)

        # Normalize contributions by maximum absolute value
        if abs(X).max() > 0:
            X = X / abs(X).max()
            assert isinstance(X, np.ndarray)

        seqlet_contrib_actual = X * O

        # Apply sign correction based on mean contribution
        unsigned_contrib = np.sign(seqlet_contrib_actual.mean()) * X

        seqlet_matrices.append(unsigned_contrib)

    return seqlets_df, seqlet_matrices


def calculate_motif_similarity(
    seqlets: list[np.ndarray],
    known_motifs: list[np.ndarray] | dict[tuple[str, str], np.ndarray],
    chunk_size: int | None = None,
    n_nearest: int | None = None,
    threshold: float | None = None,
    **kwargs,
) -> sparse.csr_array:
    """
    Calculate TomTom similarity and convert to log-space for clustering.

    Parameters
    ----------
    seqlets
        List of seqlet contribution matrices, each with shape (4, length)
    known_motifs
        List of known motif PPM matrices, each with shape (4, length)
        or a dictionary of motif PPMs, each with shape (4, length)
    chunk_size
        If provided, process seqlets in chunks of this size to manage memory usage.
        If None, process all seqlets at once (original behavior).
    n_nearest
        If provided, only keep the n most similar motifs for each seqlet.
        This creates naturally sparse matrices and reduces memory usage.
        If None, computes similarities to all motifs (with optional thresholding).
    threshold
        Similarity threshold for sparsity when n_nearest is None.
        Values below threshold are clipped to zero. Default 0.05.
        Ignored when n_nearest is specified.
    **kwargs
        Additional arguments for memelite's TomTom

    Returns
    -------
    Sparse log-transformed similarity array with shape (n_seqlets, n_motifs).
    When n_nearest is used, only the top-k similarities per seqlet are stored.
    When threshold is used, values below threshold are clipped to zero.

    Examples
    --------
    >>> _, seqlet_matrices = tfmindi.pp.extract_seqlets(contrib, oh)
    >>> # Memory-efficient: only keep top 50 similarities per seqlet
    >>> similarity_matrix = calculate_motif_similarity(seqlet_matrices, known_motifs, n_nearest=50)
    >>> print(similarity_matrix.shape)
    (1250, 3989)
    >>> # Traditional approach with thresholding
    >>> similarity_matrix = calculate_motif_similarity(seqlet_matrices, known_motifs, threshold=0.1)
    >>> # For large datasets, use chunking with n_nearest
    >>> similarity_matrix = calculate_motif_similarity(seqlet_matrices, known_motifs, chunk_size=10000, n_nearest=50)
    """
    if isinstance(known_motifs, dict):
        known_motifs = list(known_motifs.values())

    n_seqlets = len(seqlets)
    n_motifs = len(known_motifs)

    # Set default threshold if not using n_nearest
    if threshold is None and n_nearest is None:
        threshold = 0.05

    # If no chunking requested or dataset is small
    if chunk_size is None or len(seqlets) <= chunk_size:
        if n_nearest is not None:
            # Use n_nearest approach for memory efficiency
            sim, _, _, _, _, idxs = tomtom(Qs=seqlets, Ts=known_motifs, n_nearest=n_nearest, **kwargs)
            l_sim = np.nan_to_num(-np.log10(sim + 1e-10)).astype(np.float32)

            # Build sparse matrix directly from n_nearest results
            row_indices = []
            col_indices = []
            data_values = []

            for i in range(n_seqlets):
                for j in range(min(n_nearest, l_sim.shape[1])):
                    if l_sim[i, j] > 0:  # Only store positive similarities
                        row_indices.append(i)
                        col_indices.append(idxs[i, j])
                        data_values.append(l_sim[i, j])

            return sparse.csr_array(
                (data_values, (row_indices, col_indices)),
                shape=(n_seqlets, n_motifs),
                dtype=np.float32,
            )
        else:
            # Traditional full matrix approach with thresholding
            sim, _, _, _, _ = tomtom(Qs=seqlets, Ts=known_motifs, **kwargs)
            l_sim = np.nan_to_num(-np.log10(sim + 1e-10))

            # Handle empty arrays
            if l_sim.size == 0:
                return sparse.csr_array(l_sim)

            # Clip values below threshold to zero and create sparse array
            l_sim[l_sim < threshold] = 0
            return sparse.csr_array(l_sim.astype(np.float32))

    # Chunked processing - build final sparse matrix directly from coordinates
    # Collect coordinates and data for final sparse matrix construction
    row_indices = []
    col_indices = []
    data_values = []

    for i in tqdm(range(0, len(seqlets), chunk_size), desc="Processing chunks"):
        end_idx = min(i + chunk_size, len(seqlets))
        chunk = seqlets[i:end_idx]

        if n_nearest is not None:
            # Use n_nearest approach for memory efficiency
            sim_chunk, _, _, _, _, idxs_chunk = tomtom(Qs=chunk, Ts=known_motifs, n_nearest=n_nearest, **kwargs)
            l_sim_chunk = np.nan_to_num(-np.log10(sim_chunk + 1e-10)).astype(np.float32)

            # Build sparse coordinates from n_nearest results
            for local_i in range(len(chunk)):
                global_i = i + local_i
                for j in range(min(n_nearest, l_sim_chunk.shape[1])):
                    if l_sim_chunk[local_i, j] > 0:  # Only store positive similarities
                        row_indices.append(global_i)
                        col_indices.append(idxs_chunk[local_i, j])
                        data_values.append(l_sim_chunk[local_i, j])
        else:
            # Traditional thresholding approach
            sim_chunk, _, _, _, _ = tomtom(Qs=chunk, Ts=known_motifs, **kwargs)
            l_sim_chunk = np.nan_to_num(-np.log10(sim_chunk + 1e-10)).astype(np.float32)

            # Find non-zero entries above threshold
            mask = l_sim_chunk >= threshold
            if mask.any():
                chunk_rows, chunk_cols = np.where(mask)
                chunk_data = l_sim_chunk[mask]

                # Adjust row indices for global matrix position
                global_rows = chunk_rows + i

                # Accumulate coordinates and data
                row_indices.extend(global_rows)
                col_indices.extend(chunk_cols)
                data_values.extend(chunk_data)

        del sim_chunk, l_sim_chunk, chunk

    # Handle empty result
    if len(data_values) == 0:
        return sparse.csr_array((n_seqlets, n_motifs), dtype=np.float32)

    # Build final sparse matrix directly
    return sparse.csr_array(
        (data_values, (row_indices, col_indices)),
        shape=(n_seqlets, n_motifs),
        dtype=np.float32,
    )


def create_seqlet_adata(
    similarity_matrix: sparse.csr_array,
    seqlet_metadata: pd.DataFrame,
    seqlet_matrices: list[np.ndarray[Any, np.dtype[np.floating]]] | None = None,
    oh_sequences: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    contrib_scores: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    motif_names: list[str] | list[tuple[str, str]] | None = None,
    motif_collection: dict[tuple[str, str], np.ndarray[Any, np.dtype[np.floating]]]
    | list[np.ndarray[Any, np.dtype[np.floating]]]
    | None = None,
    motif_annotations: pd.DataFrame | None = None,
    motif_to_dbd: dict[str, str] | None = None,
    dtype: type[np.floating] = np.float32,
) -> AnnData:
    """
    Create comprehensive AnnData object storing all seqlet data for analysis pipeline.

    Parameters
    ----------
    similarity_matrix
        Sparse log-transformed similarity array with shape (n_seqlets, n_motifs)
    seqlet_metadata
        DataFrame with seqlet coordinates and metadata
    seqlet_matrices
        List of seqlet contribution matrices, each with shape (4, length)
    oh_sequences
        One-hot sequences for each seqlet region with shape (n_examples, 4, total_length)
    contrib_scores
        Raw contribution scores for each seqlet region with shape (n_examples, 4, total_length)
    motif_names
        List of motif names corresponding to similarity matrix columns
    motif_collection
        Dictionary or list of motif PPM matrices, each with shape (4, length)
    motif_annotations
        DataFrame with motif annotations containing TF names and other metadata
    motif_to_dbd
        Dictionary mapping motif names to DNA-binding domain annotations
    dtype
        Data type for numerical arrays to optimize memory usage (default: np.float32)

    Returns
    -------
    AnnData object with all data needed for downstream analysis

    Data Storage:

    - .X: Sparse log-transformed motif similarity array (n_seqlets × n_motifs)
    - .obs: Seqlet metadata and variable-length arrays stored per seqlet

      - Standard metadata: coordinates, attribution, scores
      - .obs["seqlet_matrix"]: Individual seqlet contribution matrices
      - .obs["seqlet_oh"]: Individual seqlet one-hot sequences
    - .obs: Additional seqlet mapping indices
      - .obs["example_oh_idx"]: Index into unique examples for one-hot sequences
      - .obs["example_contrib_idx"]: Index into unique examples for contribution scores
    - .uns: Memory-efficient storage for unique examples
      - .uns["unique_examples"]["oh"]: Unique example one-hot sequences (n_unique_examples × 4 × length)
      - .uns["unique_examples"]["contrib"]: Unique example contribution scores (n_unique_examples × 4 × length)
    - .var: Motif names and annotations
      - .var["motif_ppm"]: Individual motif PPM matrices
      - .var["dbd"]: DNA-binding domain annotations
      - .var["direct_annot"]: Direct TF annotations
      - Other annotation columns from motif_annotations DataFrame

    Examples
    --------
    >>> seqlets_df, seqlet_matrices = tm.pp.extract_seqlets(contrib, oh)
    >>> similarity_matrix = tm.pp.calculate_motif_similarity(seqlet_matrices, motifs)
    >>> adata = tm.pp.create_seqlet_adata(
    ...     similarity_matrix,
    ...     seqlets_df,
    ...     seqlet_matrices=seqlet_matrices,
    ...     oh_sequences=oh,
    ...     contrib_scores=contrib,
    ...     motif_collection=motifs,
    ...     motif_annotations=annotations,
    ...     motif_to_dbd=motif_to_dbd_dict,
    ... )
    >>> print(adata.shape)
    (295, 17995)
    """
    # Validate inputs
    n_seqlets = similarity_matrix.shape[0]  # type: ignore
    if n_seqlets != len(seqlet_metadata):
        raise ValueError(
            f"Number of seqlets in similarity matrix ({n_seqlets}) "
            f"does not match seqlet metadata ({len(seqlet_metadata)})"
        )

    if seqlet_matrices is not None and len(seqlet_matrices) != n_seqlets:
        raise ValueError(
            f"Number of seqlet matrices ({len(seqlet_matrices)}) does not match number of seqlets ({n_seqlets})"
        )

    # Create AnnData object with proper string indices
    obs_df = seqlet_metadata.copy()
    obs_df.index = obs_df.index.astype(str)

    # Create var DataFrame for motifs
    n_motifs = similarity_matrix.shape[1]  # type: ignore
    if motif_names is not None:
        if len(motif_names) != n_motifs:
            raise ValueError(
                f"Number of motif names ({len(motif_names)}) "
                f"does not match number of motifs in similarity matrix ({n_motifs})"
            )
        var_df = pd.DataFrame(index=[fn_name[1] if isinstance(fn_name, tuple) else fn_name for fn_name in motif_names])
    else:
        var_df = pd.DataFrame(index=[f"motif_{i}" for i in range(n_motifs)])

    # Store motif PPMs in .var if provided
    if motif_collection is not None:
        if isinstance(motif_collection, dict):
            motif_ppms = list(motif_collection.values())
            if motif_names is None:
                motif_names = list(motif_collection.keys())
                var_df = pd.DataFrame(
                    index=[fn_name[1] if isinstance(fn_name, tuple) else fn_name for fn_name in motif_names]
                )
        else:
            motif_ppms = motif_collection

        if len(motif_ppms) != n_motifs:
            raise ValueError(
                f"Number of motif PPMs ({len(motif_ppms)}) "
                f"does not match number of motifs in similarity matrix ({n_motifs})"
            )

        # Apply dtype conversion to motif PPMs for memory optimization
        motif_ppms_typed = [ppm.astype(dtype) for ppm in motif_ppms]
        var_df["motif_ppm"] = motif_ppms_typed  # type: ignore

    # Store motif annotations in .var if provided
    if motif_annotations is not None and motif_names is not None:
        # Add annotations for motifs that are present in the similarity matrix
        for fn_name in motif_names:
            file_name = fn_name[0] if isinstance(fn_name, tuple) else fn_name
            name = fn_name[1] if isinstance(fn_name, tuple) else fn_name
            if file_name in motif_annotations.index:
                # Add all annotation columns for this motif
                for col in motif_annotations.columns:
                    if col not in var_df.columns:
                        var_df[col] = None  # Initialize column
                    var_df.loc[name, col] = motif_annotations.loc[file_name, col]

    # Store DNA-binding domain annotations if provided
    if motif_to_dbd is not None and motif_names is not None:
        var_df["dbd"] = None  # Initialize column
        for fn_name in motif_names:
            file_name = fn_name[0] if isinstance(fn_name, tuple) else fn_name
            name = fn_name[1] if isinstance(fn_name, tuple) else fn_name
            if file_name in motif_to_dbd:
                var_df.loc[name, "dbd"] = motif_to_dbd[file_name]

    # Convert sparse array data to specified dtype for memory optimization
    if hasattr(similarity_matrix, "astype"):
        # Modern sparse arrays have astype method
        similarity_matrix_typed = similarity_matrix.astype(dtype)
    else:
        # Fallback for older sparse matrices
        similarity_matrix_typed = similarity_matrix.copy()
        similarity_matrix_typed.data = similarity_matrix_typed.data.astype(dtype)

    adata = AnnData(
        X=similarity_matrix_typed,
        obs=obs_df,
        var=var_df,
    )

    # Store seqlet-level data in .obs columns (variable length, must stay in .obs)
    if seqlet_matrices is not None and len(seqlet_matrices) > 0:
        # Apply dtype conversion and store seqlet matrices
        seqlet_matrices_typed = [matrix.astype(dtype) for matrix in seqlet_matrices]
        adata.obs["seqlet_matrix"] = seqlet_matrices_typed  # type: ignore

        # Process seqlet sequences and store unique examples
        if (oh_sequences is not None or contrib_scores is not None) and n_seqlets > 0:
            # Get unique example indices and create mapping
            unique_example_indices = seqlet_metadata["example_idx"].unique()
            example_idx_to_pos = {idx: pos for pos, idx in enumerate(unique_example_indices)}

            adata.uns["unique_examples"] = {}
            seqlet_oh_sequences: list[np.ndarray] | None = [] if oh_sequences is not None else None
            seqlet_to_example_pos_oh: list[int] | None = [] if oh_sequences is not None else None
            seqlet_to_example_pos_contrib: list[int] | None = [] if contrib_scores is not None else None

            for _, row in seqlet_metadata.iterrows():
                ex_idx = int(row["example_idx"])

                # Extract seqlet OH sequences if needed
                if (
                    oh_sequences is not None
                    and seqlet_oh_sequences is not None
                    and seqlet_to_example_pos_oh is not None
                ):
                    start = int(row["start"])
                    end = int(row["end"])
                    seqlet_oh = oh_sequences[ex_idx, :, start:end].astype(dtype)
                    seqlet_oh_sequences.append(seqlet_oh)
                    seqlet_to_example_pos_oh.append(example_idx_to_pos[ex_idx])

                # Create contrib mapping if needed
                if contrib_scores is not None and seqlet_to_example_pos_contrib is not None:
                    seqlet_to_example_pos_contrib.append(example_idx_to_pos[ex_idx])

            # Store results
            if oh_sequences is not None:
                adata.obs["seqlet_oh"] = seqlet_oh_sequences  # type: ignore
                unique_oh_sequences = oh_sequences[unique_example_indices].astype(dtype)
                adata.uns["unique_examples"]["oh"] = unique_oh_sequences
                adata.obs["example_oh_idx"] = seqlet_to_example_pos_oh  # type: ignore

            if contrib_scores is not None:
                unique_contrib_scores = contrib_scores[unique_example_indices].astype(dtype)
                adata.uns["unique_examples"]["contrib"] = unique_contrib_scores
                adata.obs["example_contrib_idx"] = seqlet_to_example_pos_contrib

    return adata


def recursive_seqlets(X, threshold=0.01, min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0, n_bins=1000):
    """Call seqlets using the recursive seqlet algorithm.

    THIS FUNCTION IS A DIRECT COPY FROM THE TANGERMEME REPOSITORY FROM JACOB SCHREIBER.
    We do a direct copy here since we only need this function and we want to avoid the heavy torch installation.

    This algorithm identifies spans of high attribution characters, called
    seqlets, using a simple approach derived from the Tomtom/FIMO algorithms.
    First, distributions of attribution sums are created for all potential
    seqlet lengths by discretizing the sum, with one set of distributions for
    positive attribution values and one for negative attribution values. Then,
    CDFs are calculated for each distribution (or, more specifically, 1-CDFs).
    Finally, p-values are calculated via lookup to these 1-CDFs for all
    potential CDFs, yielding a (n_positions, n_lengths) matrix of p-values.

    This algorithm then identifies seqlets by defining them to have a key
    property: all internal spans of a seqlet must also have been called a
    seqlet. This means that all spans from `min_seqlet_len` to `max_seqlet_len`,
    starting at any position in the seqlet, and fully contained by the borders,
    must have a p-value below the threshold. Functionally, this means finding
    entries where the upper left triangle rooted in it is comprised entirely of
    values below the threshold. Graphically, for a candidate seqlet starting at
    X and ending at Y to be called a seqlet, all the values within the bounds
    (in addition to X) must also have a p-value below the threshold.


                                                    min_seqlet_len
                                --------
    . . . . . . . | . . . . / . . . . . . . .
    . . . . . . . | . . . / . . . . . . . . .
    . . . . . . . | . . / . . . . . . . . . .
    . . . . . . . | . / . . . . . . . . . . .
    . . . . . . . | / . . . . . . . . . . . .
    . . . . . . . X . . . . . . . . Y . . . .
    . . . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . . . .


    The seqlets identified by this approach will usually be much smaller than
    those identified by the TF-MoDISco approach, including sometimes missing
    important characters on the flanks. You can set `additional_flanks` to
    a higher value if you want to include additional positions on either side.
    Importantly, the initial seqlet calls cannot overlap, but these additional
    characters are not considered when making that determination. This means
    that seqlets may appear to overlap when `additional_flanks` is set to a
    higher value.


    Parameters
    ----------
    X: np.ndarray, shape=(-1, length)
            Attributions for each position in each example. The identity of the
            characters is not relevant for seqlet calling, so this should be the
            "projected" attributions, i.e., the attribution of the observed
            characters.

    threshold: float, optional
            The p-value threshold for calling seqlets. All positions within the
            triangle (as detailed above) must be below this threshold. Default is
            0.01.

    min_seqlet_len: int, optional
            The minimum length that a seqlet must be, and the minimal length of
            span that must be identified as a seqlet in the recursive property.
            Default is 4.

    max_seqlet_len: int, optional
            The maximum length that a seqlet can be. Default is 25.

    additional_flanks: int, optional
            An additional value to subtract from the start, and to add to the end,
            of all called seqlets. Does not affect the called seqlets.
    n_bins: int, optional
        The number of bins to use when estimating the PDFs and CDFs. Default is
        1000.


    Returns
    -------
    seqlets: pandas.DataFrame, shape=(-1, 5)
            A BED-formatted dataframe containing the called seqlets, ranked from
            lowest p-value to higher p-value. The returned p-value is the p-value
            of the (location, length) span and is not influenced by the other
            values within the triangle.
    """
    columns = ["example_idx", "start", "end", "attribution", "p-value"]
    seqlets = _recursive_seqlets(X, threshold, min_seqlet_len, max_seqlet_len, additional_flanks, n_bins)
    seqlets = pd.DataFrame(seqlets, columns=columns)
    return seqlets.sort_values("p-value").reset_index(drop=True)


@numba.njit
def _recursive_seqlets(X, threshold=0.01, min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0, n_bins=1000):
    """Call seqlets recursively using the Tangermeme algorithm.

    This algorithm has four steps.

    (1) Convert attribution scores into integer bins and calculate a histogram
    (2) Convert these histograms into null distributions across lengths
    (3) Use the null distributions to calculate p-values for each possible length
    (4) Decode this matrix of p-values to find the longest seqlets
    """
    n, l = X.shape
    m = n * l

    ###
    # Step 1: Calculate a histogram of binned scores
    ###

    xmax, xmin = X.max(), X.min()
    bin_width = (xmax - xmin) / (n_bins - 1)

    f = np.zeros(n_bins, dtype=np.float64)

    for i in range(n):
        for j in range(l):
            x_bin = math.floor((X[i, j] - xmin) / bin_width)
            f[x_bin] += 1

    f = f / m

    ###
    # Step 2: Calculate null distributions across lengths
    ###

    scores = np.zeros((max_seqlet_len + 1, n_bins * max_seqlet_len), dtype=np.float64)
    scores[1, :n_bins] = f

    rcdfs = np.zeros_like(scores)
    rcdfs[:, 0] = 1.0

    for seqlet_len in range(2, max_seqlet_len + 1):
        for i in range(n_bins * (seqlet_len - 1)):
            for j in range(n_bins):
                scores[seqlet_len, i + j] += scores[seqlet_len - 1, i] * f[j]

        for i in range(1, n_bins * seqlet_len):
            rcdfs[seqlet_len, i] = max(rcdfs[seqlet_len, i - 1] - scores[seqlet_len, i], 0)

    ###
    # Step 3: Calculate p-values given these 1-CDFs
    ###

    X_csum = np.zeros((n, l + 1))
    for i in range(n):
        for j in range(l):
            X_csum[i, j + 1] = X_csum[i, j] + X[i, j]

    ###
    # Step 4: Decode p-values into seqlets
    ###

    seqlets = []

    for i in range(n):
        p_value = np.ones((max_seqlet_len + 1, l), dtype=np.float64)
        p_value[:min_seqlet_len] = 0
        p_value[:, -min_seqlet_len] = 1

        for seqlet_len in range(min_seqlet_len, max_seqlet_len + 1):
            for k in range(l - seqlet_len + 1):
                x_ = X_csum[i, k + seqlet_len] - X_csum[i, k]
                x_ = math.floor((x_ - xmin * seqlet_len) / bin_width)

                p_value[seqlet_len, k] = max(rcdfs[seqlet_len, x_], p_value[seqlet_len - 1, k])

        # Iteratively identify spans, from longest to shortest, that satisfy the
        # recursive p-value threshold.
        for j in range(max_seqlet_len - min_seqlet_len + 1):
            seqlet_len = max_seqlet_len - j

            while True:
                start = p_value[seqlet_len].argmin()
                p = p_value[seqlet_len, start]
                p_value[seqlet_len, start] = 1

                if p >= threshold:
                    break

                for k in range(1, seqlet_len):
                    if p_value[seqlet_len - k, start + k] >= threshold:
                        break

                else:
                    for end in range(start, min(start + seqlet_len, l - 1)):
                        p_value[:, end] = 1

                    end = min(start + seqlet_len + additional_flanks, l - 1)
                    start = max(start - additional_flanks, 0)
                    attr = X_csum[i, end] - X_csum[i, start]
                    seqlets.append((i, start, end, attr, p))

    return seqlets
