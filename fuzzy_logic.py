import numpy
from itertools import groupby
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from scipy import signal

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

def harmonic_clustering(fund_freq_lst, harm_freq_lst, tol=5, max_harmonic=15, max_freq=1024):
    clusters = []
    harm_freq_arr = np.array(harm_freq_lst, dtype=float)

    for f0 in fund_freq_lst:
        if f0 <= 0:
            continue

        ideal_harmonics = np.arange(1, max_harmonic + 1) * f0
        ideal_harmonics = ideal_harmonics[ideal_harmonics < max_freq]

        matched = []
        for ih in ideal_harmonics:
            diff = np.abs(harm_freq_arr - ih)
            match_idx = np.where(diff <= tol)[0]
            if len(match_idx) > 0:
                matched.append(harm_freq_arr[match_idx[0]])

        clusters.append({
            'f0': f0,
            'matched_harmonics': matched,
            'match_count': len(matched)
        })

    clusters = sorted(clusters, key=lambda x: x['match_count'], reverse=True)

    return clusters