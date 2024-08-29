import torch
import math

from torch.nn.functional import linear, normalize


class XYLoss(torch.nn.Module):
    def __init__(self,
                 num_class,
                 embedding_size,
                 s,
                 m2, # for arcface
                 m3, # for XYface
                 t,
                 errsum,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.num_class = num_class
        self.embedding_size = embedding_size
        self.s = s
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2

        # For XYLoss
        self.cos_m_ = math.cos(self.m3)
        self.sin_m_ = math.sin(self.m3)
        self.theta_ = math.cos(- self.m3)
        self.sinmm_ = math.sin( - self.m3) * self.m3

        self.easy_margin = False
        self.weight_activated = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_class, self.embedding_size)))
        self.fp16 = True
        self.t = t
        self.m_smooth = 0
        self.clear_rio = 1.0 - errsum  #(1-reesum-φ)^2 * cos(θ) noise决定了φ的上限
        self.margin_ada = True
    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(logits)
            norm_weight_activated = normalize(self.weight_activated)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]
        if self.margin_ada:
            # self.m2 = (1 - self.m_smooth) * self.m2
            self.m3 = (1 - self.m_smooth/self.clear_rio) * self.m3

            # # For ArcFace
            # self.cos_m = math.cos(self.m2)
            # self.sin_m = math.sin(self.m2)
            # self.theta = math.cos(math.pi - self.m2)
            # self.sinmm = math.sin(math.pi - self.m2) * self.m2

            # For XYLoss
            self.cos_m_ = math.cos(self.m3)
            self.sin_m_ = math.sin(self.m3)
            self.theta_ = math.cos(- self.m3)
            self.sinmm_ = math.sin(- self.m3) * self.m3

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        cos_theta_m_ = target_logit * self.cos_m_ - sin_theta * self.sin_m_  # cos(target+margin)

        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

            final_target_logit_ = torch.where(
                target_logit <= self.cos_m_, cos_theta_m_, target_logit + self.sinmm_)
        mask_hard = logits > final_target_logit.unsqueeze(1)
        mask_noise = logits > final_target_logit_.unsqueeze(1)

        hard_example = logits[mask_hard] # hard+noise
        noise_example = logits[mask_noise]

        #Recode numbers
        #num_all = 10974976 #ms1mv2 128*85742

        #num_all = self.num_class * logits.shape[0]
        num_all = 2900    #ms_celeb_1m
        easy_num = num_all - hard_example.size(0)
        noise_num = noise_example.size(0)



        # noise
        self.m_smooth = torch.div(easy_num, num_all) * 0.01 + (1 - 0.01) * self.m_smooth

        # hard
        logits[mask_hard] = hard_example * (self.t + 1)  #origin

        gamma = 2

        logits[mask_noise] = (noise_example * (1 - self.m_smooth/self.clear_rio) ** gamma).clamp_min_(1e-30)



        # target(theta+m)
        logits[index_positive, labels[index_positive].view(-1)] = final_target_logit



        logits = logits * self.s

        return logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)



        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)


        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits