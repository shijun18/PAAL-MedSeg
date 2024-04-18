import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def MyCrossEntropy(pred, target):
    log_softmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-target*log_softmax(pred),dim=1))



class CrossentropyLoss(torch.nn.CrossEntropyLoss):

    def forward(self, inp, target):
        if target.size()[1] > 1:
            target = torch.argmax(target,1)
        target = target.long()
        num_classes = inp.size()[1]

        inp = inp.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target = target.view(-1,)

        return super(CrossentropyLoss, self).forward(inp, target)




class TopKLoss(CrossentropyLoss):

    def __init__(self, weight=None, ignore_index=-100, k=10, reduction=None):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        # target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()


class DynamicTopKLoss(CrossentropyLoss):

    def __init__(self, weight=None, ignore_index=-100, step_threshold=1000, min_k=5, reduction=None):
        self.k = 100
        self.step = 0
        self.min_k = min_k
        self.step_threshold = step_threshold
        super(DynamicTopKLoss, self).__init__(weight, False, ignore_index, reduce=False)
        
    def forward(self, inp, target):
        # target = target[:, 0].long()
        res = super(DynamicTopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        self.step += 1

        if self.step % self.step_threshold == 0 and self.k > self.min_k:
            self.k -= 1
        
        return res.mean()


class OhemCELoss(nn.Module):

    def __init__(self, thresh=0.7, ignore_lb=-100):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        if len(labels.size()) > 3:
            labels = torch.argmax(labels,1)
        labels = labels.long()
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, weight=None):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight

    
    def forward(self, inp, target):
        ce = CrossentropyLoss(weight=self.weight)
        ce_loss = ce(inp,target)

        # num_classes = inp.size()[1]
        log_preds = F.log_softmax(inp, dim=1)
        smooth_loss = -log_preds.mean()

        loss = self.confidence * ce_loss + self.smoothing * smooth_loss
        return loss