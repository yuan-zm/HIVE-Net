"""Common image segmentation metrics.
"""

import torch
import numpy as np
from PIL import Image

EPS = 1e-10

def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


def _fast_hist(true, pred, num_classes):
    # pred = pred.float()
    # true = true.float()
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.

    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.

    Args:
        hist: confusion matrix.

    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.

    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.

    Args:
        hist: confusion matrix.

    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).

    Args:
        hist: confusion matrix.

    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.

    Args:
        hist: confusion matrix.

    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice


def eval_metrics(true, pred, num_classes):
    """Computes various segmentation metrics on 2D feature maps.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.

    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    # true = true.float()
    # pred = pred.float()
    # true = torch.from_numpy(true)
    pred = pred[np.newaxis, :]
    pred = torch.from_numpy(pred)

    true = true.int()
    pred = pred.int()
    hist = torch.zeros((num_classes, num_classes))
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    return overall_acc, avg_per_class_acc, avg_jacc, avg_dice


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice_coeff(pred, target):
    ims = [target, pred]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    pred = np_ims[0]
    target = np_ims[1]

    smooth = 0.

    m1 = pred.flatten() # Flatten
    m2 = target.flatten()# Flatten
    m1 = m1 / 255
    m2 = m2 / 255
    tt = m1 * m2
    intersection = (m1 * m2).sum()
    intersection = np.float(intersection)

    bing = ( np.uint8(m1) | np.uint8(m2)).sum()
    bing = np.float(bing)
    jac = intersection / bing

    ff1 = 2 * jac / (1 + jac)

    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth), jac


if __name__ == "__main__":
    # t1 = torch.rand((5, 4, 2))
    # t1 = t1 > 0.5
    # t2 = torch.rand((5, 4, 2))
    # t2 = t2 > 0.5
    # t1 = t1.int()
    # t2 = t2.int()
    # overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(t2, t1, 2)
    # print("acc:{0} , perclassacc:{1}, jcc:{2}, dice:{3}" .format(overall_acc, avg_per_class_acc, avg_jacc, avg_dice))
    t1 = torch.rand((5, 4, 2))
    t1 = t1 > 0.5
    t2 = torch.rand((5, 4, 2))
    t2 = t2 > 0.5
    t1 = t1.int()
    t2 = t2.int()
    dice = dice_coeff(t1, t2)
    print(dice)