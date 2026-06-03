"""
Wavelet + Otsu seqlet caller v2 — with boundary expansion.

Improvements over v1 (0.942):
  1. Coiflet-2 wavelet (better frequency localization than Symlet-4)
  2. Raised otsu_weight (1.3) to reduce false positives
  3. Raw-signal boundary expansion: extends interval edges by up to 2bp
     where the raw signal supports it, improving IoU at strict thresholds.

Competition-compliant:
  - Same code path for UniBind and CREsted
  - Same hyperparameters for both datasets
  - No branching on dataset identity or sequence length
  - Runtime: ~2s for all data (limit: 10s)

Train score: 0.9756 (UB=1.000, CR=0.951)
"""

import numpy as np
import pandas as pd
import pywt
import time


# ---------- helpers ----------

def _mask_to_intervals(mask):
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


def _merge_intervals(intervals, max_gap):
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def format_prediction(intervals):
    if not intervals:
        return " "
    return " ".join(f"{s} {e}" for s, e in intervals)


def compute_iou(pred_s, pred_e, gt_s, gt_e):
    inter_s = max(pred_s, gt_s)
    inter_e = min(pred_e, gt_e)
    intersection = max(0, inter_e - inter_s)
    union = (pred_e - pred_s) + (gt_e - gt_s) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def evaluate(predictions, ground_truth, dataset):
    iou_thresholds = [0.3, 0.5, 0.7]
    gt_by_seq = {}
    for _, row in ground_truth.iterrows():
        idx = row['example_idx']
        if idx not in gt_by_seq:
            gt_by_seq[idx] = []
        gt_by_seq[idx].append((row['gt_start'], row['gt_end']))

    all_recalls = []
    all_precisions = []
    for idx in sorted(gt_by_seq.keys()):
        gt_intervals = gt_by_seq[idx]
        pred_intervals = predictions.get(idx, [])
        for iou_thresh in iou_thresholds:
            matched_gt = set()
            tp = 0
            preds_sorted = sorted(pred_intervals, key=lambda x: x[0])
            for ps, pe in preds_sorted:
                for gi, (gs, ge) in enumerate(gt_intervals):
                    if gi in matched_gt:
                        continue
                    if compute_iou(ps, pe, gs, ge) >= iou_thresh:
                        matched_gt.add(gi)
                        tp += 1
                        break
            recall = tp / len(gt_intervals) if gt_intervals else 0
            precision = tp / len(preds_sorted) if preds_sorted else 0
            all_recalls.append(recall)
            if dataset == 'crested':
                all_precisions.append(precision)

    mean_recall = np.mean(all_recalls)
    if dataset == 'unibind':
        return mean_recall
    else:
        mean_precision = np.mean(all_precisions)
        if mean_precision + mean_recall == 0:
            return 0.0
        return 2 * mean_precision * mean_recall / (mean_precision + mean_recall)


# ---------- wavelet denoising ----------

def wavelet_denoise(sig, wavelet='coif2', max_level=4,
                     threshold_mode='soft', threshold_scale=1.7):
    wv = pywt.Wavelet(wavelet)
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
        denoised_coeffs.append(
            pywt.threshold(detail, value=thresh, mode=threshold_mode)
        )
    reconstructed = pywt.waverec(denoised_coeffs, wavelet)
    return reconstructed[:len(sig)]


# ---------- Otsu ----------

def otsu_threshold(sig, n_bins=256):
    sig_min, sig_max = sig.min(), sig.max()
    if sig_max - sig_min < 1e-10:
        return sig_max
    hist, bin_edges = np.histogram(sig, bins=n_bins, range=(sig_min, sig_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return sig_max
    best_thresh = sig_max
    best_variance = -1
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
    return best_thresh


# ---------- boundary refinement ----------

def _refine_boundaries_raw(intervals, raw, expand=2, contract_frac=0.2):
    """
    Refine interval boundaries using the raw |X| signal.
    Expands edges by up to `expand` positions where the raw signal
    is still above a fraction of the interval's peak.
    Contracts edges where raw signal is very weak.
    """
    result = []
    for s, e in intervals:
        peak_raw = raw[s:e].max()
        if peak_raw <= 0:
            result.append((s, e))
            continue
        cutoff = peak_raw * contract_frac
        new_s, new_e = s, e
        # Expand left
        for _ in range(expand):
            if new_s > 0 and raw[new_s - 1] > cutoff:
                new_s -= 1
            else:
                break
        # Expand right
        for _ in range(expand):
            if new_e < len(raw) and raw[new_e] > cutoff:
                new_e += 1
            else:
                break
        # Contract weak edges
        while new_s < new_e - 4 and raw[new_s] < cutoff * 0.5:
            new_s += 1
        while new_e > new_s + 4 and raw[new_e - 1] < cutoff * 0.5:
            new_e -= 1
        if new_e - new_s >= 4:
            result.append((new_s, new_e))
        elif e - s >= 4:
            result.append((s, e))
    return result


# ---------- combined caller ----------

def call_seqlets(X, wavelet='coif2', max_level=4,
                  threshold_mode='soft', threshold_scale=1.7,
                  otsu_weight=1.3, min_len=4, max_gap=2,
                  refine_expand=2, refine_contract=0.0):
    """
    Full pipeline: denoise -> threshold -> merge -> refine -> filter.

    Same code, same params for both datasets.
    """
    all_seqlets = []

    for i in range(X.shape[0]):
        raw = np.abs(X[i])

        # Wavelet denoise
        denoised = wavelet_denoise(raw, wavelet=wavelet, max_level=max_level,
                                    threshold_mode=threshold_mode,
                                    threshold_scale=threshold_scale)
        denoised = np.maximum(denoised, 0)

        # Otsu threshold
        thresh = otsu_threshold(denoised) * otsu_weight
        above = denoised > thresh
        intervals = _mask_to_intervals(above)

        # Merge small gaps
        intervals = _merge_intervals(intervals, max_gap)

        # Refine boundaries using raw signal
        if refine_expand > 0 or refine_contract > 0:
            intervals = _refine_boundaries_raw(intervals, raw,
                                                expand=refine_expand,
                                                contract_frac=refine_contract)

        # Filter too-short intervals
        intervals = [(s, e) for s, e in intervals if (e - s) >= min_len]

        all_seqlets.append(intervals)

    return all_seqlets


# ---- Main ----
if __name__ == "__main__":
    DATA_DIR = "."

    gt = pd.read_csv(f"{DATA_DIR}/train_ground_truth.csv")
    train_ub = np.load(f"{DATA_DIR}/train_unibind.npz")
    train_cr = np.load(f"{DATA_DIR}/train_crested.npz")
    test_ub = np.load(f"{DATA_DIR}/test_unibind.npz")
    test_cr = np.load(f"{DATA_DIR}/test_crested.npz")
    gt_ub = gt[gt.dataset == 'unibind']
    gt_cr = gt[gt.dataset == 'crested']

    # Best params from search
    params = dict(
        wavelet='coif2', max_level=4,
        threshold_mode='soft', threshold_scale=1.7,
        otsu_weight=1.3, min_len=4, max_gap=2,
        refine_expand=2, refine_contract=0.0,
    )

    print("=" * 60)
    print("Wavelet + Otsu v2 — evaluation and submission")
    print("=" * 60)
    print(f"Params: {params}\n")

    # Evaluate on training data
    ub_seqlets = call_seqlets(train_ub['X'], **params)
    cr_seqlets = call_seqlets(train_cr['X'], **params)

    ub_preds = {i: s for i, s in enumerate(ub_seqlets)}
    cr_preds = {i: s for i, s in enumerate(cr_seqlets)}

    ub_recall = evaluate(ub_preds, gt_ub, 'unibind')
    cr_f1 = evaluate(cr_preds, gt_cr, 'crested')
    combined = 0.5 * ub_recall + 0.5 * cr_f1

    print(f"UB_Recall={ub_recall:.4f}  CR_F1={cr_f1:.4f}  Combined={combined:.4f}")
    print(f"Baseline: 0.942  Improvement: {combined - 0.942:+.4f}\n")

    # Detailed IoU breakdown
    iou_thresholds = [0.3, 0.5, 0.7]
    for dname, preds, gt_df in [('UniBind', ub_preds, gt_ub),
                                 ('CREsted', cr_preds, gt_cr)]:
        gt_by_seq = {}
        for _, row in gt_df.iterrows():
            idx = row['example_idx']
            if idx not in gt_by_seq: gt_by_seq[idx] = []
            gt_by_seq[idx].append((row['gt_start'], row['gt_end']))

        for iou_t in iou_thresholds:
            total_tp, total_fn, total_fp = 0, 0, 0
            for idx in sorted(gt_by_seq.keys()):
                gt_intervals = gt_by_seq[idx]
                pred_intervals = preds.get(idx, [])
                matched_gt = set()
                tp = 0
                for ps, pe in sorted(pred_intervals):
                    for gi, (gs, ge) in enumerate(gt_intervals):
                        if gi in matched_gt: continue
                        if compute_iou(ps, pe, gs, ge) >= iou_t:
                            matched_gt.add(gi); tp += 1; break
                total_tp += tp
                total_fn += len(gt_intervals) - tp
                total_fp += len(pred_intervals) - tp
            r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
            print(f"  {dname} IoU>={iou_t}: TP={total_tp} FN={total_fn} FP={total_fp} "
                  f"R={r:.3f} P={p:.3f} F1={f1:.3f}")

    # Sample predictions
    print(f"\n--- Predictions vs GT ---")
    for i in range(min(10, len(ub_seqlets))):
        gt_rows = gt_ub[gt_ub.example_idx == i][['gt_start', 'gt_end']].values
        print(f"  UB {i}: pred={ub_seqlets[i]}  gt={gt_rows.tolist()}")
    print()
    for i in range(min(10, len(cr_seqlets))):
        gt_rows = gt_cr[gt_cr.example_idx == i][['gt_start', 'gt_end']].values
        print(f"  CR {i}: pred={cr_seqlets[i]}  gt={gt_rows.tolist()}")

    # Runtime check
    print(f"\n--- Runtime check ---")
    t0 = time.time()
    _ = call_seqlets(train_ub['X'], **params)
    _ = call_seqlets(train_cr['X'], **params)
    _ = call_seqlets(test_ub['X'], **params)
    _ = call_seqlets(test_cr['X'], **params)
    t_total = time.time() - t0
    print(f"  All 4 datasets: {t_total:.2f}s (limit: 10s)")

    # Generate submission
    print(f"\n=== Generating submission ===")
    ub_test = call_seqlets(test_ub['X'], **params)
    cr_test = call_seqlets(test_cr['X'], **params)

    sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    results = {}
    for i, seqs in enumerate(ub_test):
        results[f"unibind_{i}"] = format_prediction(seqs)
    for i, seqs in enumerate(cr_test):
        results[f"crested_{i}"] = format_prediction(seqs)
    sample['PredictionString'] = sample['row_id'].map(results)
    sample.to_csv(f"{DATA_DIR}/submission_wavelet_otsu_v2.csv", index=False)
    print(f"Saved submission_wavelet_otsu_v2.csv")
