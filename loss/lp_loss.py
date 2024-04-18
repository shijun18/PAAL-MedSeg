import torch
import torch.nn as nn

def LossPredLoss(input, target, reduction='mean', margin=1.0):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], batch size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already haved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


def CrossEntropy(pred, target, reduction='mean'):
    log_softmax = nn.LogSoftmax(dim=1)
    entropy = -target*log_softmax(pred) #nchw
    if reduction == 'mean':
        loss = torch.mean(torch.sum(entropy,dim=1))
    elif reduction == 'none':
        loss = torch.sum(torch.mean(entropy,dim=(2,3)),dim=1)
    else:
        NotImplementedError()
    return loss



class CEPlusLPL(nn.Module):
    
    def __init__(self, reduction=None, alpha=1.0,  **kwargs):
        super(CEPlusLPL, self).__init__()
        self.kwargs = kwargs
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, predict, target):
        predict_seg, predict_lp = predict
        assert predict_seg.size() == target.size()

        ce_loss = CrossEntropy(predict_seg, target, reduction='none') # N
        # print(torch.sum(ce_loss,dim=1).mean())
        predict_lp = predict_lp.view(predict_lp.size(0)) # N
        assert predict_lp.size() == ce_loss.size()
        lp_loss = LossPredLoss(predict_lp,ce_loss,reduction=self.reduction,**self.kwargs)
        # print(lp_loss)
        total_loss = ce_loss.mean() + self.alpha * lp_loss

        return total_loss