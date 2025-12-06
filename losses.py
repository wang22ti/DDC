import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return loss.mean()

class IBLoss(nn.Module):
    def __init__(self, num_classes, weight=None, alpha=10000.):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.num_classes = num_classes

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_classes)),1) # N * 1
        ib = grads*features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon)
        return ib_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, gpu=0):
        super(LDAMLoss, self).__init__()
        self.gpu = gpu
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).cuda(self.gpu)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.FloatTensor).cuda(self.gpu)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, x, target, use_multiplicative=True):
        output = x / self.Delta_list + self.iota_list if use_multiplicative else x + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)

class CVSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.0, tau=1.0):
        super(CVSLoss, self).__init__()
        self.cls_probs = cls_num_list / torch.sum(cls_num_list)
        self.gamma, self.tau = gamma, tau

    def forward(self, x, target, tro, kappa_multi, kappa_add):
        weight = self.cls_probs ** (- tro) if tro != 0 else None
        output = x  # 初始化输出为原始logits
        
        # 处理乘性缩放：kappa_multi可以是标量或张量
        if isinstance(kappa_multi, torch.Tensor):
            # 如果是张量，确保不包含0值以避免除零
            if torch.any(kappa_multi != 0):
                # 将kappa_multi扩展到batch维度: (num_classes,) -> (batch_size, num_classes)
                kappa_multi_expanded = kappa_multi.unsqueeze(0).expand_as(x)
                output = output * (self.cls_probs.unsqueeze(0) ** self.gamma) / kappa_multi_expanded
        elif kappa_multi != 0:
            # 如果是非零标量，使用原来的逻辑
            output = output * (self.cls_probs ** self.gamma) / kappa_multi
        
        # 处理加性调整：kappa_add可以是标量或张量
        if isinstance(kappa_add, torch.Tensor):
            # 如果是张量，确保不包含0值以避免除零
            if torch.any(kappa_add != 0):
                # 将kappa_add扩展到batch维度
                kappa_add_expanded = kappa_add.unsqueeze(0).expand_as(x)
                output = output + self.tau * torch.log(self.cls_probs.unsqueeze(0) / kappa_add_expanded)
        elif kappa_add != 0:
            # 如果是非零标量，使用原来的逻辑
            output = output + self.tau * torch.log(self.cls_probs / kappa_add)
        
        return F.cross_entropy(output, target, weight=weight)