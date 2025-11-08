import numpy
from itertools import groupby
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from scipy import signal
from magtach.op_codes.preprocess_functions import min_max_norm
from scipy.signal import find_peaks

def harmonic_voting(fund_pred, harmonic_mask, max_harm=6, w_fund=0.4, w_harm=0.6):
    L = len(fund_pred)
    score = np.zeros(L)

    for f in range(L):
        harm_sum = 0
        for k in range(2, max_harm + 1):
            idx = k * f
            if idx < L:
                harm_sum += harmonic_mask[idx]
        score[f] = w_fund * fund_pred[f] + w_harm * harm_sum

    best_bin = np.argmax(score)
    return best_bin, score



def estimate_f0_autocorr(harmonic_mask, min_period=10, max_period=512):

    # Center autocorrelation
    ac = correlate(harmonic_mask, harmonic_mask, mode='full')
    ac = ac[len(ac)//2:]                # keep positive lags
    ac[:min_period] = 0

    # plt.plot(ac)
    # plt.show()

    # Search within reasonable range
    valid_range = ac[min_period:max_period]
    lag = np.argmax(valid_range) + min_period
    f0_bin = lag                        # frequency spacing in bins
    return lag, ac

def scale_psd(orig_psd_db, final_length_psd, scale_factor, method="average"):
    if scale_factor is None or scale_factor == 1:
        return orig_psd_db[:final_length_psd]

    if method == "decimate":
        ### pick every nth bin
        cur_psd = orig_psd_db[::scale_factor]

    elif method == "nbins_average":
        orig_psd_copy = orig_psd_db[:final_length_psd * scale_factor].copy()
        cur_psd = []
        window_length = scale_factor
        for ind in range(0, len(orig_psd_copy), window_length):
            cur_psd.append(np.mean(orig_psd_copy[ind:ind + window_length]))

    elif method == "average":
        ### Average pooling sliding bins of length = scale factor
        ### no overlap
        cur_psd = orig_psd_db[:final_length_psd * scale_factor].reshape(final_length_psd, scale_factor).mean(axis=1)
    elif method == "median":
        ### Average pooling sliding bins of length = scale factor
        ### no overlap
        cur_psd = orig_psd_db[:final_length_psd * scale_factor].reshape(final_length_psd, scale_factor).median(axis=1)

    elif method == "max":
        ### Max pooling sliding bins of length = scale factor
        ### no overlap
        cur_psd = orig_psd_db[:final_length_psd * scale_factor].reshape(final_length_psd, scale_factor).max(axis=1)

    elif method == "softmax":
        blocks = orig_psd_db[:final_length_psd * scale_factor].reshape(final_length_psd, scale_factor)

        # subtract max for numerical stability
        exps = np.exp((blocks - np.max(blocks, axis=1, keepdims=True)) / 1.0)  # temperature=1.0
        weights = exps / np.sum(exps, axis=1, keepdims=True)

        cur_psd = np.sum(blocks * weights, axis=1)

    elif method == "resample":
        ### resample the psd to desired length
        cur_psd = orig_psd_db[:int(final_length_psd * scale_factor)]
        cur_psd = signal.resample(cur_psd, len(cur_psd) // scale_factor)

    else:
        raise ValueError("method must be 'decimate', 'average', or 'resample'")

    return cur_psd[:final_length_psd]

def find_windows(nd_arr):
    lst_of_1 = nd_arr
    windows_lst = []
    center_freq_lst = []

    # loop through the grouped elements along with their start indices
    start_index = 0
    for key, group in groupby(lst_of_1):
        group_list = list(group)
        group_length = len(group_list)
        if key == 1:
            end_index = start_index + group_length - 1
            center_idx = (start_index + end_index) // 2
            windows_lst.append((start_index, end_index))
            center_freq_lst.append(center_idx)
        start_index += group_length

    return windows_lst, center_freq_lst

# def harmonic_clustering(fund_freq_lst, harm_freq_lst, tolerance=5, max_harmonic=15, max_freq=1024):
#     clusters = []
#     harm_freq_arr = np.array(harm_freq_lst, dtype=float)
#
#     for f0 in fund_freq_lst:
#         if f0 <= 0 or 65>=f0>57:
#             print("f0 between 57, 65")
#             continue
#
#         ideal_harmonics = np.arange(1, max_harmonic + 1) * f0
#         ideal_harmonics = ideal_harmonics[ideal_harmonics < max_freq]
#
#         matched = []
#         for ih in ideal_harmonics:
#             diff = np.abs(harm_freq_arr - ih)
#             match_idx = np.where(diff <= tolerance)[0]
#             if len(match_idx) > 0:
#                 matched.append(harm_freq_arr[match_idx[0]])
#
#         clusters.append({
#             'f0': f0,
#             'matched_harmonics': matched,
#             'match_count': len(matched)
#         })
#
#     clusters = sorted(clusters, key=lambda x: x['match_count'], reverse=True)
#
#     return clusters

def harmonic_clustering(fund_freq_lst, harm_freq_lst, tolerance=5, max_harmonic=15, max_freq=1024):
    # ------------------------------
    # Step 0: Remove 57–67 Hz region from both lists
    # ------------------------------
    fund_freq_arr = np.array(sorted(set(fund_freq_lst)), dtype=float)
    harm_freq_arr = np.array(sorted(set(harm_freq_lst)), dtype=float)

    fund_freq_arr = fund_freq_arr[(fund_freq_arr <= 57) | (fund_freq_arr >= 67)]
    harm_freq_arr = harm_freq_arr[(harm_freq_arr <= 57) | (harm_freq_arr >= 67)]

    # use filtered arrays instead of original lists
    fund_freq_lst = fund_freq_arr
    harm_freq_lst = harm_freq_arr

    if len(fund_freq_lst) == 0:
        # print("No fund frequencies found — inferring from harmonics")
        clusters = []
        harm_freq_arr = np.array(sorted(set(harm_freq_lst)), dtype=float)

        # Remove 60 Hz interference band
        harm_freq_arr = harm_freq_arr[(harm_freq_arr <= 57) | (harm_freq_arr >= 67)]

        used_harmonics = set()

        for i, f0_candidate in enumerate(harm_freq_arr):
            if f0_candidate in used_harmonics:
                continue

            # Generate ideal harmonics
            ideal_harmonics = np.arange(1, max_harmonic + 1) * f0_candidate
            ideal_harmonics = ideal_harmonics[ideal_harmonics < max_freq]

            matched = []
            for k, ih in enumerate(ideal_harmonics, start=1):
                tol_dyn = max(tolerance, 0.02 * f0_candidate * (1 + 0.05 * (k - 1)))
                diff = np.abs(harm_freq_arr - ih)
                match_idx = np.where(diff <= tol_dyn)[0]
                if len(match_idx) > 0:
                    match_val = harm_freq_arr[match_idx[0]]
                    matched.append(match_val)
                    used_harmonics.add(match_val)

            clusters.append({
                'f0': f0_candidate,
                'matched_harmonics': matched,
                'match_count': len(matched)
            })

        clusters = sorted(clusters, key=lambda x: x['match_count'], reverse=True)
        return clusters
    else:
        clusters = []
        harm_freq_arr = np.array(harm_freq_lst, dtype=float)
        fund_freq_arr = np.array(sorted(set(fund_freq_lst)), dtype=float)

        # Step 1: Remove 60 Hz interference region
        valid_funds = []
        for f0 in fund_freq_arr:
            if f0 <= 0 or (57 < f0 <= 67):
                # print(f"Skipping f0 between 57–65 Hz: {f0}")
                continue
            valid_funds.append(f0)
        valid_funds = np.array(valid_funds, dtype=float)

        # Step 2: Remove harmonic duplicates among fund candidates (using harmonic logic)
        pruned_funds = []
        removed_funds = []

        for i, f0 in enumerate(valid_funds):
            if f0 in removed_funds:
                continue

            # Generate ideal harmonics for this f₀
            ideal_harmonics = np.arange(2, max_harmonic + 1) * f0
            ideal_harmonics = ideal_harmonics[ideal_harmonics < max_freq]

            # Eliminate other fund candidates that are harmonics of f₀
            for k, ih in enumerate(ideal_harmonics, start=2):
                # --- Dynamic tolerance scaling ---
                tol_dyn = max(tolerance, 0.02 * f0 * (1 + 0.05 * (k - 1)))
                # ---------------------------------
                for other_f0 in valid_funds:
                    if other_f0 <= f0 or other_f0 in removed_funds:
                        continue
                    if np.abs(ih - other_f0) <= tol_dyn:
                        # print(f"Removing {other_f0:.2f} Hz as harmonic of {f0:.2f} Hz [tol={tol_dyn:.2f}]")
                        removed_funds.append(other_f0)

            pruned_funds.append(f0)

        pruned_funds = np.array(sorted(set(pruned_funds)), dtype=float)
        # print(f"\nKept fundamentals: {pruned_funds}")
        # print(f"Removed (harmonic) candidates: {sorted(set(removed_funds))}\n")

        # Step 3: Perform harmonic clustering **only for pruned fundamentals**
        for f0 in pruned_funds:
            ideal_harmonics = np.arange(1, max_harmonic + 1) * f0
            ideal_harmonics = ideal_harmonics[ideal_harmonics < max_freq]

            matched = []
            for k, ih in enumerate(ideal_harmonics, start=1):
                # --- Dynamic tolerance scaling ---
                tol_dyn = max(tolerance, 0.02 * f0 * (1 + 0.05 * (k - 1)))
                # ---------------------------------
                diff = np.abs(harm_freq_arr - ih)
                match_idx = np.where(diff <= tol_dyn)[0]
                if len(match_idx) > 0:
                    matched.append(harm_freq_arr[match_idx[0]])

            clusters.append({
                'f0': f0,
                'matched_harmonics': matched,
                'match_count': len(matched)
            })

        clusters = sorted(clusters, key=lambda x: x['match_count'], reverse=True)
        return clusters


def predict_freq(orig_signal, ss_sig, strt_ind, end_ind, pred_peak):
    pred_freq = 0

    def find_peak_freq2(strt_ind, end_ind):
        fs, alpha = 44100, 30
        nfft_beta = 50
        freq_ss, Pxx_ss = signal.welch(ss_sig, fs, nperseg=len(orig_signal), nfft=int(fs * nfft_beta))

        log_Pxx_ss = np.log(Pxx_ss)
        norm_Pxx_ss = min_max_norm(log_Pxx_ss[:1024 * nfft_beta])
        norm_Pxx_ss[:10] = norm_Pxx_ss[11]

        segment = norm_Pxx_ss[(strt_ind * nfft_beta):(end_ind * nfft_beta)]
        plot_data_dump = [segment]
        # pickle.dump(plot_data_dump,open("./fine_estimate","wb"))
        p_idx = 0
        try:
            peaks = find_peaks(segment, height=(np.max(segment)) * 0.7)
            max_peak_ind = np.where(peaks[1]['peak_heights'] == np.max(peaks[1]['peak_heights']))[0][0]

            p_idx = peaks[0][max_peak_ind]
        except:
            pass
            # print("no p_idx")

        # pred_freq=p_idx/nfft_beta
        pred_freq = (freq_ss[(strt_ind * nfft_beta) + p_idx])
        return pred_freq

    pred_freq = find_peak_freq2(strt_ind, end_ind)

    return pred_freq