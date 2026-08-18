"""GPU implementation of the TOMTOM algorithm, mirroring :func:`memelite.tomtom`.

Only the first of TOMTOM's three per-query stages runs on the device. Measured against
the real 18k-motif collection, the column-distance stage is ~75% of runtime while the
p-value dynamic program is a flat per-query cost and the alignment scoring is ~12%, so
those two keep running through memelite's numba kernels on the CPU.

Two design points are load-bearing rather than incidental:

* **Layout.** ``gamma`` is held as ``(n_targets, n_query_columns)`` so that the reduction
  kernels can give one thread to each query column and still have neighbouring threads
  touch neighbouring addresses.
* **Batching.** The reductions accumulate strictly sequentially over targets, which is
  what makes them bit-identical to the CPU *and* free of atomics. Their parallelism is
  therefore the number of query columns in flight, so a batch of one query cannot fill a
  GPU and a batch of a few hundred can.

The distance kernel writes its four products as separate subtractions rather than a
single dot product, matching the CPU's rounding, and avoids cuBLAS entirely so that
summation order does not vary between GPU architectures.
"""

from __future__ import annotations

import math

import numba
import numpy as np

# memelite's per-query kernels for the two stages that stay on the CPU.
from memelite.tomtom import (  # type: ignore
    _merge_rc_results,
    _p_value_backgrounds,
    _p_values,
)
from numba import njit, prange

_gamma_kernel = None
_module = None


def _compile():
    """Compile the device kernels on first use and cache them at module scope."""
    global _gamma_kernel, _module
    if _module is not None:
        return
    import cupy as cp  # type: ignore

    _gamma_kernel = cp.ElementwiseKernel(
        "raw float64 X, raw float64 Y, raw float64 Xn, raw float64 Yn, int64 nqtot, int64 nt, int64 ncol, int64 c0",
        "float64 g",
        """
        long long j = i / ncol, c = i - j * ncol;
        long long xi = c0 + c;
        double z = Xn[xi] + Yn[j];
        z -= 2.0 * X[          xi] * Y[       j];
        z -= 2.0 * X[  nqtot + xi] * Y[  nt + j];
        z -= 2.0 * X[2*nqtot + xi] * Y[2*nt + j];
        z -= 2.0 * X[3*nqtot + xi] * Y[3*nt + j];
        g = z > 0.0 ? -sqrt(z) : 0.0;
        """,
        "tfmindi_gamma",
    )

    _module = cp.RawModule(
        code=r"""
extern "C" {

/* Bin masses for the median approximation. The added values are integer target
   multiplicities, so int64 accumulation is exact whatever order the atomics land in --
   the CPU stores the same integers in a float64 array. */
__global__ void median_mass(const double* __restrict__ gamma,
                            const double* __restrict__ zmin,
                            const double* __restrict__ span,
                            const long long* __restrict__ counts,
                            unsigned long long* mass,
                            long long ncol, long long nt, int n_median_bins)
{
    long long total = nt * ncol;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        long long j = idx / ncol, c = idx - j * ncol;
        int b = (int)((gamma[idx] - zmin[c]) / span[c] * (n_median_bins - 1));
        atomicAdd(&mass[c * n_median_bins + b], (unsigned long long)counts[j]);
    }
}

/* The crossing bin's value sum: real floating-point accumulation, so it is walked in
   target order by a single thread per query column, exactly as the CPU does. */
__global__ void median_value(const double* __restrict__ gamma,
                             const double* __restrict__ zmin,
                             const double* __restrict__ span,
                             const long long* __restrict__ counts,
                             const long long* __restrict__ kbin,
                             double* value,
                             long long ncol, long long nt, int n_median_bins)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    int k = (int)kbin[c];
    double zc = zmin[c], sc = span[c], s = 0.0;
    for (long long j = 0; j < nt; ++j) {
        double g = gamma[j * ncol + c];
        int b = (int)((g - zc) / sc * (n_median_bins - 1));
        if (b == k) s += g * (double)counts[j];
    }
    value[c] = s;
}

/* Integerize into gamma_int and build the score histogram, again one thread per query
   column walking targets in order into a private accumulator -- no atomics, and the
   same term-by-term division the CPU performs. gamma_int is written per query as
   (n_targets, Q_max) with the query axis reversed, which is the layout _p_values reads. */
__global__ void integerize(const double* __restrict__ gamma,
                           const double* __restrict__ medians,
                           const long long* __restrict__ counts,
                           const long long* __restrict__ qidx,
                           const long long* __restrict__ lpos,
                           const long long* __restrict__ qlen,
                           const long long* __restrict__ bin_scale,
                           const long long* __restrict__ offset,
                           double* f, signed char* gamma_int,
                           long long ncol, long long nt, long long q_max,
                           int n_bins, double ys)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;

    long long q = qidx[c], l = lpos[c], nq = qlen[q];
    long long bs = bin_scale[q], off = offset[q];
    double m = medians[c];

    double acc[101];
    for (int b = 0; b <= n_bins; ++b) acc[b] = 0.0;

    signed char* out = gamma_int + q * nt * q_max + (nq - 1 - l);
    for (long long j = 0; j < nt; ++j) {
        double g = gamma[j * ncol + c];
        long long x = (long long)floor((g - m) * (double)bs + 0.5);
        out[j * q_max] = (signed char)(x - off);
        acc[x] += (double)counts[j] / ys;
    }
    double* frow = f + q * q_max * (n_bins + 1) + l * (n_bins + 1);
    for (int b = 0; b <= n_bins; ++b) frow[b] = acc[b];
}

}
""",
        options=("-std=c++11",),
    )


@njit(parallel=True, cache=True)
def _stages_23(
    gamma_int,
    f_all,
    offsets,
    Q_lens,
    T_lens,
    rr_inv,
    n_score_bins,
    n_cache,
    T_max,
    Q_max,
    n_threads,
    reverse_complement,
    n_nearest,
    n_in_targets,
    n_out_targets,
    n_outputs,
):
    """Run the p-value background DP and the alignment scoring for a batch of queries.

    This is memelite's ``_tomtom`` parallel loop with its first stage removed: the
    integerized scores and score histograms arrive already computed. Per-thread
    scratchpads are allocated once and reused, as upstream does.
    """
    n_len = Q_max * n_score_bins + Q_max * n_cache
    _A = np.empty((n_threads, Q_max, Q_max, n_len), dtype=np.float64)
    _A_csum = np.empty((n_threads, Q_max, Q_max, n_len), dtype=np.float64)
    _B = np.empty((n_threads, T_max + 1, n_len), dtype=np.float64)
    _results = np.empty((n_threads, len(T_lens), 5), dtype=np.float64)
    results = np.empty((len(Q_lens), n_out_targets, n_outputs), dtype=np.float64)

    for b in prange(len(Q_lens)):
        nq = Q_lens[b]
        pid = numba.get_thread_id()
        offset = offsets[b]

        _p_value_backgrounds(f_all[b], _A[pid], _B[pid], _A_csum[pid], nq, n_score_bins, T_max, offset)
        _p_values(gamma_int[b], _B[pid], rr_inv, T_lens, -1, nq, offset, _results[pid])

        if reverse_complement == 1:
            _merge_rc_results(_results[pid])
        else:
            _results[pid, :, 4] = 0

        if n_nearest == -1:
            results[b] = _results[pid, :n_in_targets]
        else:
            idxs = np.argsort(_results[pid, :n_in_targets, 0])[:n_nearest]
            results[b, :, :5] = _results[pid, idxs]
            results[b, :, 5] = idxs

    return results


def _prepare(Qs, Ts, n_target_bins, reverse_complement):
    """Replicate memelite's host-side preprocessing: concatenation and target binning."""
    Q_lens = np.array([q.shape[-1] for q in Qs], dtype="int64")
    Q = np.concatenate(Qs, axis=-1)
    Q_norm = (Q**2).sum(axis=0)

    if reverse_complement:
        Ts = list(Ts) + [T[::-1, ::-1] for T in Ts]

    T_lens = np.array([t.shape[-1] for t in Ts], dtype="int64")
    T = np.concatenate(Ts, axis=-1)
    T_norm = (T**2).sum(axis=0)

    if Q_norm.max() == 0 or T_norm.max() == 0:
        raise ValueError("Cannot have all-zeroes as targets or query.")

    if n_target_bins is not None:
        T_min = T.min(axis=-1, keepdims=True)
        T_max = T.max(axis=-1, keepdims=True)
        T_max[T_max == T_min] = T_min[T_max == T_min] + 1

        T_ints = np.around((T - T_min) / (T_max - T_min) * (n_target_bins - 1))
        T_ints = T_ints.T.dot(n_target_bins ** np.arange(len(T))[:, None])
        _, rr_idxs, rr_inv, rr_counts = np.unique(
            T_ints.flatten(), return_index=True, return_inverse=True, return_counts=True
        )
        T = T[:, rr_idxs]
        T_norm = T_norm[rr_idxs]
        rr_inv = rr_inv.astype("uint64")
    else:
        rr_inv = np.arange(T.shape[-1]).astype("uint64")
        rr_counts = np.ones_like(rr_inv)

    return Q, T, Q_lens, T_lens, Q_norm, T_norm, rr_inv, rr_counts.astype("int64")


def _batch_size(nt, q_max, n_queries, fraction=0.5):
    """Choose how many queries to hold on the device at once, from free GPU memory.

    ``gamma`` dominates: one float64 per (target column, query column). ``gamma_int``
    adds a byte per (query, target column, Q_max). Sized from what the card actually has
    free rather than a constant, so small cards iterate more instead of failing.
    """
    import cupy as cp  # type: ignore

    free, _ = cp.cuda.Device().mem_info
    per_query = q_max * nt * 8 + nt * q_max  # gamma + gamma_int, worst case
    return max(1, min(int(n_queries), int(free * fraction / per_query)))


def _stage1_batch(
    Xd, Yd, Xnd, Ynd, counts_d, c0, ncol, qidx, lpos, qlen_b, n_bins, n_median_bins, n_cache, q_max, total_counts
):
    """Compute integerized scores, score histograms and offsets for one batch.

    Returns device arrays ``(gamma_int, f, offsets)`` shaped ``(n_q, nt, q_max)``,
    ``(n_q, q_max, n_bins + 1)`` and ``(n_q,)``.
    """
    import cupy as cp  # type: ignore

    _compile()
    nt = Yd.shape[-1]
    n_q = len(qlen_b)

    gamma = cp.empty((nt, ncol), dtype=cp.float64)
    _gamma_kernel(Xd, Yd, Xnd, Ynd, Xd.shape[-1], nt, ncol, c0, gamma)

    z_min_ = gamma.min(axis=0)
    z_max_ = gamma.max(axis=0)
    span = z_max_ - z_min_

    mass = cp.zeros((ncol, n_median_bins), dtype=cp.uint64)
    block = 256
    grid = (min((nt * ncol + block - 1) // block, 8192),)
    _module.get_function("median_mass")(grid, (block,), (gamma, z_min_, span, counts_d, mass, ncol, nt, n_median_bins))

    # `count >= total / 2` without the float division the CPU uses.
    k = (2 * cp.cumsum(mass.astype(cp.int64), axis=1) >= total_counts).argmax(axis=1)
    value = cp.empty(ncol, dtype=cp.float64)
    cgrid = ((int(ncol) + 127) // 128,)
    _module.get_function("median_value")(
        cgrid, (128,), (gamma, z_min_, span, counts_d, k.astype(cp.int64), value, ncol, nt, n_median_bins)
    )
    medians = value / mass[cp.arange(ncol), k].astype(cp.float64)

    # Per-query scale factors. Only a few thousand values, so the segmented reduction and
    # the floor() arithmetic are cheaper and clearer to do on the host.
    zmin_h = cp.asnumpy(z_min_ - medians)
    zmax_h = cp.asnumpy(z_max_ - medians)
    med_h = cp.asnumpy(medians)
    qidx_h = cp.asnumpy(qidx)

    i_min = np.empty(n_q, dtype="int64")
    bin_scale = np.empty(n_q, dtype="int64")
    for q in range(n_q):
        sel = qidx_h == q
        im = int(math.floor(zmin_h[sel].min()))
        i_min[q] = im
        bin_scale[q] = int(math.floor(n_bins / (zmax_h[sel].max() - im)))
    offsets = -i_min * bin_scale
    med_h = med_h + i_min[qidx_h]

    if (offsets > n_cache).any():
        raise ValueError(f"Offset {int(offsets.max())} is larger than n_cache={n_cache}; increase n_cache.")

    gamma_int = cp.empty((n_q, nt, q_max), dtype=cp.int8)
    f = cp.zeros((n_q, q_max, n_bins + 1), dtype=cp.float64)
    _module.get_function("integerize")(
        cgrid,
        (128,),
        (
            gamma,
            cp.asarray(med_h),
            counts_d,
            qidx,
            lpos,
            qlen_b,
            cp.asarray(bin_scale),
            cp.asarray(offsets),
            f,
            gamma_int,
            ncol,
            nt,
            q_max,
            n_bins,
            float(total_counts),
        ),
    )
    del gamma, mass, value
    return gamma_int, f, offsets


def gpu_tomtom(
    Qs,
    Ts,
    n_nearest=None,
    n_score_bins=100,
    n_median_bins=1000,
    n_target_bins=100,
    n_cache=100,
    reverse_complement=True,
    n_jobs=-1,
):
    """Assign p-values to motif similarity, with the distance stage on the GPU.

    A drop-in replacement for :func:`memelite.tomtom` with the same arguments and the
    same return signature. See that function for the meaning of each parameter.

    Returns
    -------
    The same stacked array of best p-values, scores, offsets, overlaps and strands that
    :func:`memelite.tomtom` returns, plus target indices when ``n_nearest`` is set.
    """
    import cupy as cp  # type: ignore

    _compile()

    if n_jobs != -1:
        _n_jobs = numba.get_num_threads()
        numba.set_num_threads(n_jobs)
    else:
        n_jobs = _n_jobs = numba.get_num_threads()

    if n_nearest is None:
        n_nearest = -1
    if not isinstance(Qs[0], np.ndarray):
        Qs = [Q.numpy() for Q in Qs]
    if not isinstance(Ts[0], np.ndarray):
        Ts = [T.numpy() for T in Ts]

    Q, T, Q_lens, T_lens, Q_norm, T_norm, rr_inv, rr_counts = _prepare(Qs, Ts, n_target_bins, reverse_complement)
    nt = T.shape[-1]
    q_max = int(Q_lens.max())
    t_max = int(T_lens.max())
    n_in_targets = len(T_lens) // 2 if reverse_complement else len(T_lens)
    n_out_targets = n_in_targets if n_nearest == -1 else n_nearest
    n_outputs = 5 if n_nearest == -1 else 6
    total_counts = int(rr_counts.sum())

    Xd, Yd = cp.asarray(Q), cp.asarray(T)
    Xnd, Ynd = cp.asarray(Q_norm), cp.asarray(T_norm)
    counts_d = cp.asarray(rr_counts)

    q_off = np.zeros(len(Q_lens) + 1, dtype="int64")
    q_off[1:] = np.cumsum(Q_lens)

    step = _batch_size(nt, q_max, len(Q_lens))
    out = np.empty((len(Q_lens), n_out_targets, n_outputs), dtype="float64")

    for lo in range(0, len(Q_lens), step):
        hi = min(lo + step, len(Q_lens))
        qlen_b = Q_lens[lo:hi]
        c0, c1 = int(q_off[lo]), int(q_off[hi])
        ncol = c1 - c0

        qidx = np.repeat(np.arange(hi - lo, dtype="int64"), qlen_b)
        lpos = np.concatenate([np.arange(n, dtype="int64") for n in qlen_b])

        gamma_int, f, offsets = _stage1_batch(
            Xd,
            Yd,
            Xnd,
            Ynd,
            counts_d,
            c0,
            ncol,
            cp.asarray(qidx),
            cp.asarray(lpos),
            cp.asarray(qlen_b),
            n_score_bins,
            n_median_bins,
            n_cache,
            q_max,
            total_counts,
        )
        gi_h = cp.asnumpy(gamma_int)
        f_h = cp.asnumpy(f)
        del gamma_int, f

        out[lo:hi] = _stages_23(
            gi_h,
            f_h,
            offsets,
            qlen_b,
            T_lens,
            rr_inv,
            n_score_bins,
            n_cache,
            t_max,
            q_max,
            n_jobs,
            int(reverse_complement),
            n_nearest,
            n_in_targets,
            n_out_targets,
            n_outputs,
        )

    if n_jobs != -1:
        numba.set_num_threads(_n_jobs)

    return out.transpose(2, 0, 1)
