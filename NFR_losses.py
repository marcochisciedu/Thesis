import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Baselines losses to reduce Negative Flip Rate (NFR) while training a new model
"""
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
    
class ContrastiveFeaturesLoss(nn.Module):
    def __init__(self, tau):
        super(ContrastiveFeaturesLoss, self).__init__()
        self.tau = tau
    
    def forward(self, 
                feat_new, 
                feat_old, 
                labels, 
                new_cls_num, 
                old_cls_num,
                only_old=True
               ):
        loss = self._loss(feat_old, feat_new, labels, new_cls_num, old_cls_num, only_old)
        return loss

    def _loss(self, feat_old, feat_new, labels, new_cls_num, old_cls_num, only_old):
        """Calculates infoNCE loss on the features extracted by the old and new model.
            The embedding size should be the same
        Args:
            feat_old:
                features extracted with the old model.
                Shape: (batch_size, embedding_size)
            feat_new:
                features extracted with the new model.
                Shape: (batch_size, embedding_size)
            labels:
                Labels of the images.
                Shape: (batch_size,) 
            new_cls_num:
                Number of classes of the new model
            old_cls_num:
                Number of classes of the old model
            only_old: 
                Use only features of images that belong to the old classes
        Returns:
            Mean loss over the mini-batch.
        """
        # Select only features of images that belong to the old classes
        if (old_cls_num != new_cls_num) & only_old:
            mask = labels < min(new_cls_num, old_cls_num) # mask samples belong to new classes
            labels = labels[mask]
            feat_new = feat_new[mask] # align output dim
            feat_old = feat_old[mask]
        
        ## Create diagonal mask that only selects similarities between
        ## representations of the same images
        batch_size = feat_old.shape[0]
        diag_mask = torch.eye(batch_size, device=feat_old.device, dtype=torch.bool)
        sim_01 = torch.einsum("nc,mc->nm", feat_old, feat_new) *  self.tau

        positive_loss = -sim_01[diag_mask]
        # Get the labels of feat_old and feat_new samples
        labels_0 = labels.unsqueeze(1).expand(-1, batch_size)  # Shape: (batch_size, batch_size)
        labels_1 = labels.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, batch_size)

        # Mask similarities between the same class
        class_mask = labels_0 == labels_1
        sim_01 = (sim_01* (~class_mask)).view(batch_size, -1)   

        negative_loss_01 = torch.logsumexp(sim_01, dim=1)
        return (positive_loss + negative_loss_01).mean()
    
class ContrastivePrototypeLoss(nn.Module):
    def __init__(self, tau):
        super(ContrastivePrototypeLoss, self).__init__()
        self.tau = tau
    
    def forward(self, 
                prototype_old, 
                prototype_new):
        loss = self._loss(prototype_old, prototype_new)
        return loss

    def _loss(self, prototype_old, prototype_new):
        """Calculates infoNCE loss on the class prototypes of the old and new model.

        Args:
            prototype_old:
                class prototypes of the old model.
                Shape: (num_old_prototypes, embedding_size)
            prototype_new:
                class prototypes of the new model.
                Shape: (num_new_prototypes, embedding_size)                   
        """
        # Select only the class prototypes that both models share
        if prototype_old.shape[0] != prototype_new.shape[0]:
            prototype_old = prototype_old[:min(prototype_old.shape[0], prototype_new.shape[0])]
            prototype_new = prototype_new[:min(prototype_old.shape[0], prototype_new.shape[0])]

        ## create diagonal mask that only selects similarities between
        ## the same class prototype
        num_prototypes = prototype_old.shape[0]
        diag_mask = torch.eye(num_prototypes, device=prototype_old.device, dtype=torch.bool)
        sim_01 = torch.einsum("nc,mc->nm", prototype_old, prototype_new) *  self.tau
        positive_loss = -sim_01[diag_mask]
        # Get the other class prototypes
        sim_01 = (sim_01* (~diag_mask)).view(num_prototypes, -1)
        negative_loss_01 = torch.logsumexp(sim_01, dim=1)
        return (positive_loss + negative_loss_01).mean()
    
class CosinePrototypeLoss(nn.Module):
    def __init__(self):
        super(CosinePrototypeLoss, self).__init__()
        self.cosine_loss  = nn.CosineEmbeddingLoss()
    def forward(self, 
                prototype_old, 
                prototype_new):
        loss = self._loss(prototype_old, prototype_new)
        return loss

    def _loss(self, prototype_old, prototype_new):
        """Calculates infoNCE loss on the class prototypes of the old and new model.

        Args:
            prototype_old:
                class prototypes of the old model.
                Shape: (num_old_prototypes, embedding_size)
            prototype_new:
                class prototypes of the new model.
                Shape: (num_new_prototypes, embedding_size)                   
        """
        # Select only the class prototypes that both models share
        if prototype_old.shape[0] != prototype_new.shape[0]:
            prototype_old = prototype_old[:min(prototype_old.shape[0], prototype_new.shape[0])]
            prototype_new = prototype_new[:min(prototype_old.shape[0], prototype_new.shape[0])]

        cosine_prototype_loss = self.cosine_loss(prototype_old, prototype_new, torch.ones(prototype_old.size(0)).cuda())
        
        return cosine_prototype_loss
    
def self_cosine_distances(input):
    norm = torch.norm(input, 2, 1, True)
    norm_input = torch.div(input, norm)
    cosine_sim_matrix= torch.mm(norm_input, norm_input.transpose(0,1))
    return 1-cosine_sim_matrix

class CosineDifferencePrototypeLoss(nn.Module):
    def __init__(self):
        super(CosineDifferencePrototypeLoss, self).__init__()
    def forward(self, 
                prototype_old, 
                prototype_new):
        loss = self._loss(prototype_old, prototype_new)
        return loss

    def _loss(self, prototype_old, prototype_new):
        """Calculates infoNCE loss on the class prototypes of the old and new model.

        Args:
            prototype_old:
                class prototypes of the old model.
                Shape: (num_old_prototypes, embedding_size)
            prototype_new:
                class prototypes of the new model.
                Shape: (num_new_prototypes, embedding_size)                   
        """
        # Select only the class prototypes that both models share
        if prototype_old.shape[0] != prototype_new.shape[0]:
            prototype_old = prototype_old[:min(prototype_old.shape[0], prototype_new.shape[0])]
            prototype_new = prototype_new[:min(prototype_old.shape[0], prototype_new.shape[0])]

        # Calculate distances between prototype of the old and new model
        self_dist_old = self_cosine_distances(prototype_old) 
        self_dist_new = self_cosine_distances(prototype_new) 

        difference_distance = torch.abs(self_dist_old- self_dist_new)
        return torch.sum(difference_distance)
    
class ProximityAwareCrossEntropyLoss(nn.Module):
    def __init__(self, knn_matrix, lambda_pa):
        super(ProximityAwareCrossEntropyLoss, self).__init__()
        self.knn_matrix = knn_matrix
        self.lambda_pa = lambda_pa
    def forward(self,
                logits,
                targets):
        loss = self._loss(logits, targets)
        return loss

    def _loss(self, logits, targets):
        """Calculates Cross Entropy Loss with a higher penalty for misclassification between 
        classes that are near each other

        Args:
            logits:
                output of the model
                Shape: (batch_size, num_classes)
            targets:
                ground truth of each image
                Shape: (batch_size)                   
        """

        # Gather class Knn for each sample
        knn = self.knn_matrix[targets]  # Shape: (B, C)

        # Compute penalty weights
        weights =  1 #+ self.lambda_pa * knn  # Shape: (B, C)

        lp = logits - ((logits.exp() * weights) + 1e-09).sum(-1).log().unsqueeze(-1)
        loss_ce = F.nll_loss(lp, targets)
        
        return loss_ce