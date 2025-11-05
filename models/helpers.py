from scipy import signal
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

# cur_scaled_psd = scale_psd(resamp_log_Pxx_ss, psd_length, scale_factor,
#                                                                method="average")