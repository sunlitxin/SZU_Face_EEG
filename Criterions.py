import torch
import torch.nn as nn

class NewCrossEntropy(nn.Module):
    def __init__(self):
        super(NewCrossEntropy, self).__init__()

    def forward(self, logits, label):
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        #distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        # local to global
        logits = logits - max_logits  #logits.sub(max_logits)
        logits = torch.exp(logits)  #logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        #distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        # local to global
        logits = torch.div(logits, sum_logits_exp) #logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1,device=logits.device)
        loss[index] = logits[index].gather(1, label[index].unsqueeze(1))
        #distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        # ctx.save_for_backward(index, logits, label)
        return loss.clamp_min(1e-30).log().mean() * (-1)