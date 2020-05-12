import torch
import torch.nn as nn

class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def forward(self, logits, true, eps=1e-7):
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
            probas = torch.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))

        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)


class extend_mse_loss(nn.Module):
    """
    this loss is implement of "efficient and robust cell detection: A structured regression approach"
    """
    def __init__(self, beta, lamda):
        super(extend_mse_loss, self).__init__()
        self.beta = beta
        self.lamada = lamda

    def forward(self, pred, mask, eps=1e-7):
        batch, depth, hight, width = pred.shape
        pix_num = depth * hight * width
        y_i = torch.sum(mask) / pix_num
        first_term = self.beta * mask + self.lamada * y_i
        extend_mse_loss = 0.5 * torch.sum((pred - mask) ** 2 * first_term) / pix_num
        return extend_mse_loss


class jacc_loss(nn.Module):
    def __init__(self):
        super(jacc_loss, self).__init__()

    def forward(self, logits, true, eps=1e-7):
        """Computes the Jaccard loss, a.k.a the IoU loss.
            Note that PyTorch optimizers minimize a loss. In this
            case, we would like to maximize the jaccard loss so we
            return the negated jaccard loss.
            Args:
                true: a tensor of shape [B, H, W] or [B, 1, H, W].
                logits: a tensor of shape [B, C, H, W]. Corresponds to
                    the raw output or logits of the model.
                eps: added to the denominator for numerical stability.
            Returns:
                jacc_loss: the Jaccard loss.
            """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
            probas = torch.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return 1 - jacc_loss


if __name__ == "__main__":
    t = jacc_loss()
    t1 = torch.Tensor([[1,1],[1,0]]).long()
    t2 = torch.Tensor([[1,0],[1,0]]).long()
    t(t1,t2)
