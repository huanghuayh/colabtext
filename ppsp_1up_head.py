import torch
import torch.nn as nn
import torch.nn.functional as F
from fpn_2 import PPSP

class PPSP_withFundamental(nn.Module):
    def __init__(self, pretrained_ppsp, freeze=True, hidden=256):
        super().__init__()
        self.harm_net = pretrained_ppsp
        self.freeze = freeze

        if freeze:
            for p in self.harm_net.parameters():
                p.requires_grad = False
            print("Frozen PPSP backbone weights")

        # Dense head for the fundamental task
        self.fund_head = nn.Sequential(
            nn.Linear(1024, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1024)
        )

    def forward(self, x):
        # If backbone is frozen, force it into eval mode and stop gradients
        if self.freeze:
            self.harm_net.eval()
            with torch.no_grad():
                harm_pred = self.harm_net(x)  # (B,1,1024)
        else:
            harm_pred = self.harm_net(x)  # trainable backbone

        # Dense fundamental head (trainable)
        h = harm_pred.squeeze(1)               # (B,1024)
        fund_pred = self.fund_head(h).unsqueeze(1)  # (B,1,1024)

        return fund_pred, harm_pred





# ppsp_weights_file = "best_model_weights_fan5_fan3_bldc_fpn2"
#
# ppsp_model = PPSP(in_channels=1)
# ppsp_model.load_state_dict(torch.load(f'../../data/train_test_data/{ppsp_weights_file}', map_location="cpu"))

#
# model = PPSP_withFundamental(backbone, freeze=True, hidden=256)
#
# x = torch.randn(2, 1, 1024)
# fund_pred, harm_pred = model(x)
# print(f"fund_pred: {fund_pred.shape}, harm_pred: {harm_pred.shape}")