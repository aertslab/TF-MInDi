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
    _p_value_backgrounds,
)
from numba import njit, prange

_gamma_kernel = None
_module = None

#: Compile-time bound on the per-thread alignment window in the p-value kernel. Covers
#: ``max(target length) + max(query length) - 1``; longer motifs fall back to the CPU.
_TSUMS_MAX = 256


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

/* Best alignment and its p-value, one thread per (query, target). The running window
   sums live in a private array, so there are no atomics and nothing depends on thread
   ordering -- this kernel is deterministic on any card by construction. */
__global__ void p_values(const signed char* __restrict__ gamma_int,
                         const double* __restrict__ B,
                         const unsigned long long* __restrict__ rr_inv,
                         const long long* __restrict__ T_lens,
                         const long long* __restrict__ T_offsets,
                         const long long* __restrict__ Q_lens,
                         const long long* __restrict__ offsets,
                         double* results,
                         long long n_q, long long n_targets, long long nt_u,
                         long long q_max, long long t_max, long long n_len)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_q * n_targets) return;
    long long q = tid / n_targets, i = tid - q * n_targets;

    long long nq = Q_lens[q], nt = T_lens[i], off = offsets[q];
    const signed char* g = gamma_int + q * nt_u * q_max;
    const double* Bq = B + q * (t_max + 1) * n_len;

    short t_sums[TSUMS_MAX];
    long long m = nt + nq - 1;
    short base = (short)(nq * off);
    for (long long k = 0; k < m; ++k) t_sums[k] = base;

    long long toff = T_offsets[i];
    for (long long k = 0; k < nt; ++k) {
        const signed char* gk = g + (long long)rr_inv[toff + k] * q_max;
        for (long long l = 0; l < nq; ++l) t_sums[k + l] += gk[l];
    }

    /* memelite leaves the offset/overlap slots stale before the first hit; a score of
       exactly zero at k = 0 needs nq == 1 or offset == 0, so the first position always
       wins in practice and the tie-break below only ever sees values it set itself.
       The comparison really is stored-offset against overlap, as upstream writes it. */
    double best_p = 1.0;
    long long best_score = 0, best_off = 0, best_ov = 0;
    bool have = false;
    for (long long k = 0; k < m; ++k) {
        long long score = t_sums[k];
        long long lo = (k + 1 < nq) ? (k + 1) : nq;
        long long hi = (k - nt + 1 > 0) ? (k - nt + 1) : 0;
        long long overlap = lo - hi;
        if (score >= best_score) {
            if (have && score == best_score && best_off >= overlap) continue;
            best_p = Bq[nt * n_len + (score - 1)];
            best_score = score;
            best_off = k - nq + 1;
            best_ov = overlap;
            have = true;
        }
    }
    double* r = results + (q * n_targets + i) * 5;
    r[0] = best_p; r[1] = (double)best_score;
    r[2] = (double)best_off; r[3] = (double)best_ov; r[4] = 0.0;
}

/* Take the better of the two strands, one thread per (query, forward target). */
__global__ void merge_rc(double* results, long long n_q, long long n_targets)
{
    long long n = n_targets / 2;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_q * n) return;
    long long q = tid / n, i = tid - q * n;

    double* a = results + (q * n_targets + i) * 5;
    double* b = results + (q * n_targets + i + n) * 5;

    /* Spelled with round-to-nearest intrinsics so the compiler cannot contract
       1 - (1-p)*(1-p) into an FMA: the single rounding that would save differs from
       numpy's two by one ULP, which is enough to reorder tied p-values in the
       n_nearest ranking. */
    double pmin = a[0] < b[0] ? a[0] : b[0];
    double om = __dsub_rn(1.0, pmin);
    a[0] = __dsub_rn(1.0, __dmul_rn(om, om));
    a[4] = 0.0;
    if (a[1] <= b[1]) { a[1] = b[1]; a[2] = b[2]; a[3] = b[3]; a[4] = 1.0; }
}

}
""",
        options=("-std=c++11", f"-DTSUMS_MAX={_TSUMS_MAX}"),
    )


@njit(parallel=True, cache=True)
def _stage2_dp(f_all, offsets, Q_lens, n_score_bins, n_cache, T_max, Q_max, n_threads):
    """Build the p-value background CDFs for a batch of queries.

    This is the one stage that stays on the CPU: the dynamic program costs a flat
    ~1.4 s per query regardless of how many motifs are being scored against, so its
    share shrinks as the collection grows, and its ~48 MB per-query workspace is what
    makes a device port awkward.

    Returns
    -------
    Array of shape ``(n_queries, T_max + 1, n_len)`` holding each query's background
    survival probabilities.
    """
    n_len = Q_max * n_score_bins + Q_max * n_cache
    n_q = len(Q_lens)
    _A = np.empty((n_threads, Q_max, Q_max, n_len), dtype=np.float64)
    _A_csum = np.empty((n_threads, Q_max, Q_max, n_len), dtype=np.float64)
    B_all = np.empty((n_q, T_max + 1, n_len), dtype=np.float64)

    for b in prange(n_q):
        pid = numba.get_thread_id()
        _p_value_backgrounds(f_all[b], _A[pid], B_all[b], _A_csum[pid], Q_lens[b],
                             n_score_bins, T_max, offsets[b])
    return B_all


@njit(cache=True)
def _select_nearest(merged, out, n_nearest):
    """Keep each query's ``n_nearest`` best targets, ranked by p-value.

    Compiled rather than run through numpy because memelite selects inside its own
    ``@njit`` loop: the two sorts order tied p-values differently, and ties are common
    enough that a host-side ``np.argsort`` picks a different set of motifs.
    """
    for b in range(merged.shape[0]):
        idxs = np.argsort(merged[b, :, 0])[:n_nearest]
        out[b, :, :5] = merged[b][idxs]
        out[b, :, 5] = idxs


def _stage3_gpu(gamma_int, B_all, rr_inv_d, T_lens_d, T_offsets_d, Q_lens_d, offsets_d,
                n_q, n_targets, nt_u, q_max, t_max, n_len, reverse_complement):
    """Score every alignment and merge strands on the device.

    Returns a host array of shape ``(n_q, n_in_targets, 5)`` -- p-value, score, offset,
    overlap and strand per (query, target).
    """
    import cupy as cp  # type: ignore

    results = cp.empty((n_q, n_targets, 5), dtype=cp.float64)
    B_d = cp.asarray(B_all)
    block = 128
    total = n_q * n_targets
    _module.get_function("p_values")(
        ((total + block - 1) // block,), (block,),
        (gamma_int, B_d, rr_inv_d, T_lens_d, T_offsets_d, Q_lens_d, offsets_d, results,
         n_q, n_targets, nt_u, q_max, t_max, n_len),
    )
    del B_d

    n_in = n_targets // 2 if reverse_complement else n_targets
    if reverse_complement:
        half = n_q * n_in
        _module.get_function("merge_rc")(
            ((half + block - 1) // block,), (block,), (results, n_q, n_targets)
        )
    return cp.asnumpy(results[:, :n_in])


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


def _batch_size(nt, q_max, t_max, n_len, n_targets, n_queries, fraction=0.4):
    """Choose how many queries to hold on the device at once, from free GPU memory.

    ``gamma`` dominates, at one float64 per (target column, query column); the
    background CDFs, the integerized scores and the result table add the rest. Sized
    from what the card actually has free rather than a constant, so small cards iterate
    more instead of failing.
    """
    import cupy as cp  # type: ignore

    free, _ = cp.cuda.Device().mem_info
    per_query = (
        q_max * nt * 8          # gamma
        + nt * q_max            # gamma_int
        + (t_max + 1) * n_len * 8  # background CDFs
        + n_targets * 5 * 8     # results
    )
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

    n_len = q_max * n_score_bins + q_max * n_cache
    if t_max + q_max - 1 > _TSUMS_MAX:
        raise ValueError(
            f"Alignment window {t_max + q_max - 1} exceeds the kernel bound "
            f"{_TSUMS_MAX}; motifs this long need the CPU path."
        )

    q_off = np.zeros(len(Q_lens) + 1, dtype="int64")
    q_off[1:] = np.cumsum(Q_lens)
    t_off = np.zeros(len(T_lens), dtype="int64")
    t_off[1:] = np.cumsum(T_lens)[:-1]

    rr_inv_d = cp.asarray(rr_inv)
    T_lens_d = cp.asarray(T_lens)
    T_offsets_d = cp.asarray(t_off)

    step = _batch_size(nt, q_max, t_max, n_len, len(T_lens), len(Q_lens))
    out = np.empty((len(Q_lens), n_out_targets, n_outputs), dtype="float64")

    for lo in range(0, len(Q_lens), step):
        hi = min(lo + step, len(Q_lens))
        qlen_b = Q_lens[lo:hi]
        c0, c1 = int(q_off[lo]), int(q_off[hi])
        ncol = c1 - c0
        n_b = hi - lo

        qidx = np.repeat(np.arange(n_b, dtype="int64"), qlen_b)
        lpos = np.concatenate([np.arange(n, dtype="int64") for n in qlen_b])

        gamma_int, f, offsets = _stage1_batch(
            Xd, Yd, Xnd, Ynd, counts_d, c0, ncol,
            cp.asarray(qidx), cp.asarray(lpos), cp.asarray(qlen_b),
            n_score_bins, n_median_bins, n_cache, q_max, total_counts,
        )
        # Only the score histograms come back to the host; the integerized scores stay
        # on the device and feed the scoring kernel directly.
        f_h = cp.asnumpy(f)
        del f

        B_all = _stage2_dp(f_h, offsets, qlen_b, n_score_bins, n_cache,
                           t_max, q_max, n_jobs)

        merged = _stage3_gpu(
            gamma_int, B_all, rr_inv_d, T_lens_d, T_offsets_d,
            cp.asarray(qlen_b), cp.asarray(offsets),
            n_b, len(T_lens), nt, q_max, t_max, n_len, bool(reverse_complement),
        )
        del gamma_int, B_all

        if n_nearest == -1:
            out[lo:hi] = merged
        else:
            _select_nearest(merged, out[lo:hi], n_nearest)

    if n_jobs != -1:
        numba.set_num_threads(_n_jobs)

    return out.transpose(2, 0, 1)
