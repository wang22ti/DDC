import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, logit, target):
        return focal_loss(F.cross_entropy(logit, target, reduction='none', weight=self.weight), self.gamma)


class WeightedCELoss(nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        self.cls_probs = torch.Tensor(cls_num_list) / torch.sum(torch.Tensor(cls_num_list))
        
    def forward(self, logit, target, tro=None):
        weight = self.cls_probs ** (- tro) if tro != 0 and tro is not None else None
        if weight is not None:
            weight = weight.to(logit.device)
        
        return F.cross_entropy(logit, target, weight=weight)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (max_m / torch.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s
        self.cls_num_list = cls_num_list

    def forward(self, logit, target, tro):
        index = torch.zeros_like(logit, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        logit_m = logit - batch_m * self.s  # scale only the margin, as the logit is already scaled.

        output = torch.where(index, logit_m, logit)
        
        if tro is None:
            weight = None
        else:
            effective_num = 1.0 - torch.pow(tro, self.cls_num_list)
            weight = (1.0 - tro) / effective_num
            weight = weight / torch.sum(weight) * len(self.cls_num_list)
            weight = weight.to(logit.device).float()

        # print('weight:', weight)
        return F.cross_entropy(output, target, weight=weight)

class ClassBalancedLoss(nn.Module):
    def __init__(self, cls_num_list, beta=0.9999):
        super().__init__()
        per_cls_weights = (1.0 - beta) / (1.0 - (beta ** cls_num_list))
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class GeneralizedReweightLoss(nn.Module):
    def __init__(self, cls_num_list, exp_scale=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        per_cls_weights = 1.0 / (cls_num_ratio ** exp_scale)
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num

    def forward(self, logit, target):
        logit_adjusted = logit + self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)


class LogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau

    def forward(self, logit, target):
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)


class LADELoss(nn.Module):
    def __init__(self, cls_num_list, remine_lambda=0.1, estim_loss_weight=0.1):
        super().__init__()
        self.num_classes = len(cls_num_list)
        self.prior = cls_num_list / torch.sum(cls_num_list)

        self.balanced_prior = torch.tensor(1. / self.num_classes).float().to(self.prior.device)
        self.remine_lambda = remine_lambda

        self.cls_weight = cls_num_list / torch.sum(cls_num_list)
        self.estim_loss_weight = estim_loss_weight

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)
 
        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, logit, target):
        logit_adjusted = logit + torch.log(self.prior).unsqueeze(0)
        ce_loss =  F.cross_entropy(logit_adjusted, target)

        per_cls_pred_spread = logit.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (logit - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)
        estim_loss = -torch.sum(estim_loss * self.cls_weight)

        return ce_loss + self.estim_loss_weight * estim_loss

class CVSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.0, tau=1.0):
        super(CVSLoss, self).__init__()
        self.cls_probs = cls_num_list / torch.sum(cls_num_list)
        self.gamma, self.tau = gamma, tau

    def forward(self, x, target, tro, kappa_multi, kappa_add):
        weight = self.cls_probs ** (- tro) if tro != 0 else None
        output = x  # 初始化输出为原始logits
        
        # 处理乘性缩放：如果kappa_multi不为0（可以是标量或张量）
        if isinstance(kappa_multi, torch.Tensor):
            # 确保张量不包含0值以避免除零
            if torch.any(kappa_multi != 0):
                output = output * (self.cls_probs ** self.gamma) / kappa_multi
        elif kappa_multi != 0:
            # 如果是非零标量，使用原来的逻辑
            output = output * (self.cls_probs ** self.gamma) / kappa_multi
        
        # 处理加性调整：如果kappa_add不为0（可以是标量或张量）
        if isinstance(kappa_add, torch.Tensor):
            # 确保张量不包含0值以避免除零
            if torch.any(kappa_add != 0):
                output = output + self.tau * torch.log(self.cls_probs / kappa_add)
        elif kappa_add != 0:
            # 如果是非零标量，使用原来的逻辑
            output = output + self.tau * torch.log(self.cls_probs / kappa_add)
        
        return F.cross_entropy(output, target, weight=weight)

class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None):
        super(VSLoss, self).__init__()

        # 确保cls_num_list是numpy数组或python列表
        if isinstance(cls_num_list, torch.Tensor):
            cls_num_list = cls_num_list.cpu().numpy()
        elif not isinstance(cls_num_list, (list, np.ndarray)):
            cls_num_list = np.array(cls_num_list)

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        
        # 存储类别概率，用于动态权重计算
        self.cls_probs = torch.Tensor(cls_probs)
        
        self.weight = weight
        self.use_multiplicative = True

    def forward(self, x, target, tro=None):
        output = x / self.Delta_list + self.iota_list if self.use_multiplicative else x + self.iota_list

        # 动态计算权重（参照CE损失的方式）
        if tro is not None and tro != 0:
            weight = self.cls_probs ** (- tro)
            weight = weight.to(x.device)
        else:
            weight = self.weight

        return F.cross_entropy(output, target, weight=weight)