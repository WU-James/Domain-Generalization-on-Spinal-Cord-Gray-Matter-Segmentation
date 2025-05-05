import torch
from medpy.metric.binary import dc, jc, asd, hd
from torch import nn
import numpy as np

# Self-defined Loss
class DiceLoss(nn.Module):
    def __init__(self):
        """Simple constructor for the class."""
        super(DiceLoss, self).__init__()

    def dice_loss(self, predicted, target, ep=1e-8):
        intersection = 2 * torch.sum(predicted * target) + ep
        union = torch.sum(predicted) + torch.sum(target) + ep
        loss = 1 - intersection / union

        return loss

    def batch_dice_loss(self, predicted, target):
        batch_loss = 0.0
        for i in range(predicted.size(0)):
            batch_loss += self.dice_loss(predicted[i], target[i])
        return batch_loss / predicted.size(0)

    def forward(self, predicted, target):
        """ Method for calculation of combined loss from sample."""
        return self.batch_dice_loss(predicted, target)


# Evaluation Metrics
def compute_dice(output_list, spinal_cord_mask_list_trian, gm_mask_list_trian):
    dice_score1 = 0.0
    dice_score2 = 0.0

    for i in range(len(output_list)):
        dice_score1 += dc(output_list[i][0, :, :], spinal_cord_mask_list_trian[i].squeeze())
        dice_score2 += dc(output_list[i][1, :, :], gm_mask_list_trian[i].squeeze())

    dice_score1 /= len(output_list)
    dice_score2 /= len(output_list)

    return dice_score1, dice_score2


def compute_jaccard(output_list, spinal_cord_mask_list_trian, gm_mask_list_trian):
    jaccard_score1 = 0.0
    jaccard_score2 = 0.0

    for i in range(len(output_list)):
        jaccard_score1 += jc(output_list[i][0, :, :], spinal_cord_mask_list_trian[i].squeeze())
        jaccard_score2 += jc(output_list[i][1, :, :], gm_mask_list_trian[i].squeeze())

    jaccard_score1 /= len(output_list)
    jaccard_score2 /= len(output_list)

    return jaccard_score1, jaccard_score2


def compute_asd(output_list, spinal_cord_mask_list_trian, gm_mask_list_trian):
    asd_score1 = 0.0
    asd_score2 = 0.0

    for i in range(len(output_list)):
        sum1 = output_list[i][0, :, :].sum()
        sum2 = spinal_cord_mask_list_trian[i].squeeze().sum()
        sum3 = output_list[i][1, :, :].sum()
        sum4 = gm_mask_list_trian[i].squeeze().sum()

        if (sum1 * sum2 == 0):
            if (sum1 + sum2 == 0):
                asd_score1 += 0
            else:
                asd_score1 += np.size(output_list[i][0, :, :], 0)
        else:
            asd_score1 += asd(output_list[i][0, :, :], spinal_cord_mask_list_trian[i].squeeze())

        if (sum3 * sum4 == 0):
            if (sum3 + sum4 == 0):
                asd_score2 += 0
            else:
                asd_score2 += np.size(output_list[i][1, :, :], 0)
        else:
            asd_score2 += asd(output_list[i][1, :, :], gm_mask_list_trian[i].squeeze())

    asd_score1 /= len(output_list)
    asd_score2 /= len(output_list)

    return asd_score1, asd_score2


def compute_hd(output_list, spinal_cord_mask_list_trian, gm_mask_list_trian):
    hd_score1 = 0.0
    hd_score2 = 0.0

    for i in range(len(output_list)):
        sum1 = output_list[i][0, :, :].sum()
        sum2 = spinal_cord_mask_list_trian[i].squeeze().sum()
        sum3 = output_list[i][1, :, :].sum()
        sum4 = gm_mask_list_trian[i].squeeze().sum()

        if (sum1 * sum2 == 0):
            if (sum1 + sum2 == 0):
                hd_score1 += 0
            else:
                hd_score1 += np.size(output_list[i][0, :, :], 0)
        else:
            hd_score1 += hd(output_list[i][0, :, :], spinal_cord_mask_list_trian[i].squeeze())

        if (sum3 * sum4 == 0):
            if (sum3 + sum4 == 0):
                hd_score2 += 0
            else:
                hd_score2 += np.size(output_list[i][1, :, :], 0)
        else:
            hd_score2 += hd(output_list[i][1, :, :], gm_mask_list_trian[i].squeeze())

    hd_score1 /= len(output_list)
    hd_score2 /= len(output_list)

    return hd_score1, hd_score2
