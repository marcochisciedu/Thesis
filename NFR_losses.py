import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalDistillationLoss(nn.Module):
    """
    This class implements the focal distillation loss for PC-training.

    It requires three input data:
    - New model's prediction
    - Old model's prediction
    - groundtruth

    It supports two types of Distillation loss
    - Focal distillation with the KL-divergence loss (default)
    - Focal distillation with L2 loss

    There are two parameters adjusting the focal weights
    - fd_alpha: the weight of the background samples (default = 1)
    - fd_beta: the weight of the focused samples (default = 0)

    fd_alpha = 1 and fd_beta = 0 resembles the normal knowledge distillation setup

    This loss function is meant to be used as a additional loss term to a classification loss function (e.g. CrossEntropy)

    """

    def __init__(self, fd_alpha=1, fd_beta=5,
                 focus_type='old_correct',
                 distillation_type='kl',
                 kl_temperature=100,
                 ):
        super(FocalDistillationLoss, self).__init__()

        self._fd_alpha = fd_alpha
        self._fd_beta = fd_beta

        self._fd_type = focus_type
        self._distill_type = distillation_type

        if distillation_type == 'kl':
            self.distill_loss = nn.KLDivLoss(reduction='none')
        elif distillation_type == 'l2':
            self.distill_loss = nn.MSELoss(reduction='none')
        else:
            raise ValueError("Unknown loss type: {}".format(self._distill_type))

        self._kl_temperature = kl_temperature

    def forward(self,
                new_model_prediction, old_model_prediction,
                gt):

        old_cls_num = old_model_prediction.size(1)
        new_cls_num = new_model_prediction.size(1)
        if old_cls_num != new_cls_num:
            #TODO: may generate empty tensor, need to be fixed
            mask = gt < min(new_cls_num, old_cls_num) # mask samples belong to new classes
            gt = gt[mask]
            new_model_prediction = new_model_prediction[mask, :min(new_cls_num, old_cls_num)] # align output dim
            old_model_prediction = old_model_prediction[mask, :min(new_cls_num, old_cls_num)]

        # get old and new model prediction
        old_model_correct = old_model_prediction.argmax(dim=1) == gt
        new_model_correct = new_model_prediction.argmax(dim=1) == gt

        # loss weights in Positive-congruent training paper
        if self._fd_type == 'old_correct':
            loss_weights = old_model_correct.int()[:, None] * self._fd_beta + self._fd_alpha
        # loss weights that prioritize old model when it is correct and the new model is not
        elif self._fd_type == 'neg_flip':
            loss_weights = (old_model_correct & (~new_model_correct)).int().unsqueeze(1) * self._fd_beta + self._fd_alpha
        # loss weights that follow MUSCLE's improved NFR, always prioritize old model when the new is incorrect
        elif self._fd_type == 'new_incorrect':
            loss_weights = (~new_model_correct).int().unsqueeze(1) * self._fd_beta + self._fd_alpha
        else:
            raise ValueError("Unknown focus type: {}".format(self._fd_type))

        # get per-sample loss
        if self._distill_type == 'kl':
            sample_loss = self.distill_loss(
                F.log_softmax(new_model_prediction / self._kl_temperature, dim=1),
                F.softmax(old_model_prediction / self._kl_temperature, dim=1)).sum(dim=1) * (self._kl_temperature ** 2)
        elif self._distill_type == 'l2':
            sample_loss = self.distill_loss(new_model_prediction, old_model_prediction)
        else:
            raise ValueError("Unknown loss type: {}".format(self._distill_type))

        # weighted sum of losses
        return (sample_loss * loss_weights).mean()