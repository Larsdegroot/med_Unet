import torch
import torch.nn as nn
from scipy.ndimage import label

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss
    


class F1Score(nn.Module):
    def __init__(self, smooth=1e-6):
        super(F1Score, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        tp = torch.sum(y_pred * y_true)  # True Positives
        fp = torch.sum(y_pred * (1 - y_true))  # False Positives
        fn = torch.sum((1 - y_pred) * y_true)  # False Negatives

        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)

        f1_score = 2 * (precision * recall) / (precision + recall + self.smooth)
        return f1_score



class LesionSensitivity(nn.Module):
    def __init__(self, smooth=1e-6):
        super(LesionSensitivity, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Ensure binary masks
        y_pred = (y_pred > 0.5).int()  # Convert to binary predictions
        y_true = (y_true > 0.5).int()  # Convert to binary ground truth

        # Flatten the masks to simplify calculations
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)

        # Calculate intersection and union for sensitivity
        true_positives = (y_pred_flat * y_true_flat).sum()  # Count of true positives
        total_true = y_true_flat.sum()  # Count of true lesions (ground truth)

        # Sensitivity = TP / (TP + FN)
        sensitivity = (true_positives + self.smooth) / (total_true + self.smooth)
        return sensitivity
