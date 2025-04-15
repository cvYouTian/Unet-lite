import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth = 1.0):
    """
    clac the dice loss
    Args:
        pred: result of prediction
        target: result of target

    Returns: dice loss
    """

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()

    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """

    target = torch.cat((target, target), dim=1)
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss

