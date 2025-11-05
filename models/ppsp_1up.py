"""
both ppsp and 1up head are trained together
1) call ppsp_backbone from fpn2
2) final output from fpn2 -> all harmonics head (trainable)
3) create new head similar to ppsp_1up_head that is (trainable)
"""
import torch.nn as nn
import torch
from scipy import signal
import numpy as np


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

class PPSP_1up(nn.Module):
    def __init__(self, ppsp_backbone, hidden_nodes=256):
        super(PPSP_1up, self).__init__()

        self.ppsp_backbone = ppsp_backbone
        self.scale_fuse = nn.Conv1d(3, 1, 1)
        self.fund_head = nn.Sequential(
            nn.Linear(1024, hidden_nodes),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_nodes, 1024)
        )

    def forward(self, x):
        x = self.ppsp_backbone(x)
        harmonics_pred = x.squeeze(1)

        remade_harmonics_pred = []
        for ind in range(harmonics_pred.shape[0]):
            cur_scaled_lst=[]
            for sf in [1, 2, 3]:
                cur_scaled = scale_psd(harmonics_pred[ind],
                                int(len(harmonics_pred[ind]) // sf),
                                sf, method="average")
                # cur_scaled = torch.from_numpy(cur_scaled).float().to(x.device)
                cur_scaled = torch.nn.functional.interpolate(cur_scaled.unsqueeze(0).unsqueeze(0),
                                                      size=harmonics_pred.shape[-1],
                                                      mode="linear", align_corners=False).squeeze()
                cur_scaled_lst.append(cur_scaled)

            # flatten = torch.cat(cur_scaled_lst)
            stacked = torch.stack(cur_scaled_lst, dim=0).unsqueeze(0)
            fused = self.scale_fuse(stacked).squeeze()
            remade_harmonics_pred.append(fused)
        remade_harmonics_pred = torch.stack(remade_harmonics_pred)

        fundamental_pred = self.fund_head(remade_harmonics_pred).unsqueeze(1)

        return fundamental_pred, harmonics_pred


# import fpn_2
# import torch
#
# x= torch.randn(10, 1, 1024)
# PPSP_net = fpn_2.PPSP(in_channels=1, out_channels=32)
# model_ppsp_1up = PPSP_1up(PPSP_net, hidden_nodes=256)
# fund_pred,harmonic_pred =model_ppsp_1up(x)
# print()
