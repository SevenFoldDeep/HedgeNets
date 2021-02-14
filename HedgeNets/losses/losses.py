import torch
import torch.nn.functional as F
import numpy as np

def dice_loss(pred, truth):

    truth = F.one_hot(truth, num_classes = pred.shape[1]).permute(0,3,1,2).contiguous()
    loss = 1-((2*truth*pred + 1) / (truth + pred + 1))

    return torch.mean(loss)

def binary_focal_loss(pred, truth, gamma=2., alpha=.25):
    eps = 1e-8
    pred = nn.Softmax(1)(pred)

    truth = F.one_hot(truth, num_classes = pred.shape[1]).permute(0,3,1,2).contiguous()

    pt = torch.where(truth == 1, pred, 1-pred)

    pt = torch.clamp(pt, eps, 1. - eps)

    return (-torch.mean(alpha * torch.pow(1. - pt[:, 1, :, :], gamma) * torch.log(pt[:, 1, :, :])) - torch.mean((1 - alpha) * torch.pow(1-pt[:, 0, :, :], gamma) * torch.log(pt[:, 0, :, :])))/2
