"""
both ppsp and 1up head are trained together
1) call ppsp_backbone from fpn2
2) final output from fpn2 -> all harmonics head (trainable)
3) create new head similar to ppsp_1up_head that is (trainable)
"""
import torch.nn as nn
from helpers import scale_psd
import torch

class PPSP_1up(nn.Module):
    def __init__(self, ppsp_backbone, hidden_nodes=256):
        super(PPSP_1up, self).__init__()

        self.ppsp_backbone = ppsp_backbone

        self.fund_head = nn.Sequential(
            nn.Linear(1877, hidden_nodes),
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
            for scale_factor in [1,2,3]:
                cur_scaled = scale_psd(harmonics_pred[ind],int(len(harmonics_pred[ind])//scale_factor),scale_factor,method="average")
                cur_scaled_lst.append(cur_scaled)
            flatten = torch.cat(cur_scaled_lst)
            remade_harmonics_pred.append(flatten)
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
