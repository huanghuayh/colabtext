### Initialize model, loss, optimizer
import torch.nn as nn
import torch

class OverlapDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions).float()
        targets = targets.float()

        # compute per-sample statistics
        TP = (predictions * targets).sum(dim=1)
        FP = (predictions * (1 - targets)).sum(dim=1)
        FN = ((1 - predictions) * targets).sum(dim=1)

        # F1 / Dice formulation
        dice_per_sample = (2 * TP + self.smooth) / ((2 * TP) + FP + FN + self.smooth)
        loss = 1 - dice_per_sample

        # mean over batch
        return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, w_dice=0.5, pos_weight=None, device='cuda'):
        super().__init__()
        self.w_dice = w_dice
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.dice = OverlapDiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return (1 - self.w_dice) * bce_loss + self.w_dice * dice_loss