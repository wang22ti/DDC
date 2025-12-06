"""
CVS Manager - 管理CVS损失函数的动态参数计算
"""
import torch
import numpy as np
from collections import deque


class CVSScalingManager:
    """
    CVS缩放因子管理器
    负责维护memory bank并动态计算kappa_multi和kappa_add参数
    """
    
    def __init__(self, num_classes, cls_num_list, samples_per_class=1000, device='cuda'):
        """
        初始化CVS缩放管理器
        
        Args:
            num_classes (int): 类别数量
            cls_num_list (list): 每个类别的样本数量
            samples_per_class (int): 每个类别在memory bank中保存的最大样本数
            device (str): 设备
        """
        self.num_classes = num_classes
        self.cls_num_list = cls_num_list
        self.samples_per_class = samples_per_class
        self.device = device
        
        # 初始化类别memory bank
        self._init_class_memory_bank()
        
        # 初始化缩放因子（两种模式）
        self.class_scale_factors_logits = torch.ones(num_classes, device=device)  # 替换kappa_multi
        self.class_scale_factors_softmax = torch.ones(num_classes, device=device)  # 替换kappa_add
        
    def _init_class_memory_bank(self):
        """初始化类别memory bank，为每个类别维护一个固定大小的FIFO队列"""
        self.class_memory_bank = {}
        
        # 为每个类别初始化空的memory bank
        for class_idx in range(self.num_classes):
            # 对于样本数少于默认值的类别，使用其实际样本数作为容量
            capacity = min(self.samples_per_class, self.cls_num_list[class_idx])
            self.class_memory_bank[class_idx] = {
                'logits': deque(maxlen=capacity),
                'labels': deque(maxlen=capacity),
                'count': 0,
                'capacity': capacity
            }
        
        # 统计调整情况
        adjusted_classes = sum(1 for i in range(self.num_classes) 
                              if self.cls_num_list[i] < self.samples_per_class)
        
        print(f"[CVSManager] 初始化类别memory bank，默认每个类别最多保存 {self.samples_per_class} 个样本")
        if adjusted_classes > 0:
            print(f"[CVSManager] {adjusted_classes} 个类别的容量已根据实际样本数调整")
    
    def update_memory_bank(self, logits, labels):
        """
        更新类别memory bank，使用FIFO策略
        
        Args:
            logits (torch.Tensor): 当前batch的logits
            labels (torch.Tensor): 当前batch的labels
        """
        # 确保在CPU上处理以节省GPU内存
        logits_cpu = logits.detach().cpu()
        labels_cpu = labels.cpu()
        
        # 按类别组织数据
        for class_idx in range(self.num_classes):
            class_mask = labels_cpu == class_idx
            if class_mask.sum() > 0:
                class_logits = logits_cpu[class_mask]
                class_labels = labels_cpu[class_mask]
                
                bank = self.class_memory_bank[class_idx]
                
                # 添加新样本（deque会自动处理FIFO）
                for i in range(class_logits.size(0)):
                    bank['logits'].append(class_logits[i])
                    bank['labels'].append(class_labels[i])
                
                # 更新计数
                bank['count'] = len(bank['logits'])
    
    def get_memory_bank_data(self):
        """
        从memory bank中获取所有数据用于缩放因子拟合
        
        Returns:
            tuple: (logits, labels) 合并的tensor
        """
        all_logits = []
        all_labels = []
        
        for class_idx in range(self.num_classes):
            bank = self.class_memory_bank[class_idx]
            if bank['count'] > 0:
                # 将deque转换为tensor
                class_logits = torch.stack(list(bank['logits']))
                class_labels = torch.stack(list(bank['labels']))
                
                all_logits.append(class_logits)
                all_labels.append(class_labels)
        
        if len(all_logits) == 0:
            return None, None
        
        # 合并所有类别的数据
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return logits, labels
    
    def print_memory_bank_stats(self):
        """打印memory bank的统计信息"""
        total_samples = 0
        non_empty_classes = 0
        full_classes = 0
        adjusted_classes = 0
        
        print("[CVSManager] Memory Bank 统计信息:")
        for class_idx in range(min(10, self.num_classes)):
            bank = self.class_memory_bank[class_idx]
            total_samples += bank['count']
            if bank['count'] > 0:
                non_empty_classes += 1
            if bank['count'] == bank['capacity']:
                full_classes += 1
            if bank['capacity'] < self.samples_per_class:
                adjusted_classes += 1
            print(f"  类别 {class_idx}: {bank['count']}/{bank['capacity']} 样本")
        
        if self.num_classes > 10:
            # 统计剩余类别
            for class_idx in range(10, self.num_classes):
                bank = self.class_memory_bank[class_idx]
                total_samples += bank['count']
                if bank['count'] > 0:
                    non_empty_classes += 1
                if bank['count'] == bank['capacity']:
                    full_classes += 1
                if bank['capacity'] < self.samples_per_class:
                    adjusted_classes += 1
            print(f"  ... (还有 {non_empty_classes - min(10, non_empty_classes)} 个非空类别)")
        
        print(f"  总样本数: {total_samples}, 非空类别数: {non_empty_classes}/{self.num_classes}")
        print(f"  已满类别数: {full_classes}, 调整大小类别数: {adjusted_classes}")
        if non_empty_classes > 0:
            print(f"  平均每类样本数: {total_samples / non_empty_classes:.1f}")
    
    def fit_scaling_factors(self, n_bins=20, min_scale_factor=0.1, max_iter=1000, 
                          lr=0.01, use_logits=True):
        """
        拟合类别缩放因子
        
        Args:
            n_bins (int): ECE计算的bin数量
            min_scale_factor (float): 最小缩放因子
            max_iter (int): 最大迭代次数
            lr (float): 学习率
            use_logits (bool): True使用logits模式（替换kappa_multi），False使用softmax模式（替换kappa_add）
        
        Returns:
            torch.Tensor: 类别缩放因子
        """
        # 从memory bank获取数据
        logits, labels = self.get_memory_bank_data()
        
        if logits is None:
            print(f"[CVSManager] Warning: Memory bank为空，返回默认缩放因子")
            return torch.ones(self.num_classes, device=self.device)
        
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        
        # 初始化缩放因子
        scale_factors = torch.ones(self.num_classes, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([scale_factors], lr=lr)
        
        mode_name = "logits" if use_logits else "softmax"
        print(f"[CVSManager] 开始拟合{mode_name}模式缩放因子...")
        
        best_ece = float('inf')
        best_scale_factors = scale_factors.clone().detach()
        patience = 50
        no_improve_count = 0
        
        for iter_idx in range(max_iter):
            optimizer.zero_grad()
            
            # 应用缩放因子
            if use_logits:
                # Logits模式：直接缩放logits
                scaled_logits = logits * scale_factors.unsqueeze(0)
            else:
                # Softmax模式：在softmax之后调整
                probs = torch.softmax(logits, dim=1)
                scaled_logits = torch.log(probs * scale_factors.unsqueeze(0) + 1e-10)
            
            # 计算ECE
            ece = self._compute_ece(scaled_logits, labels, n_bins)
            
            # 反向传播
            ece.backward()
            optimizer.step()
            
            # 限制缩放因子范围
            with torch.no_grad():
                scale_factors.clamp_(min=min_scale_factor, max=10.0)
            
            # 记录最佳结果
            current_ece = ece.item()
            if current_ece < best_ece:
                best_ece = current_ece
                best_scale_factors = scale_factors.clone().detach()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 早停
            if no_improve_count >= patience:
                print(f"[CVSManager] 早停于迭代 {iter_idx}, 最佳ECE: {best_ece:.6f}")
                break
            
            # 打印进度
            if (iter_idx + 1) % 100 == 0:
                print(f"[CVSManager] 迭代 {iter_idx+1}/{max_iter}, ECE: {current_ece:.6f}, "
                      f"范围: [{scale_factors.min().item():.4f}, {scale_factors.max().item():.4f}]")
        
        print(f"[CVSManager] {mode_name}模式拟合完成, 最终ECE: {best_ece:.6f}")
        print(f"[CVSManager] 缩放因子范围: [{best_scale_factors.min().item():.4f}, {best_scale_factors.max().item():.4f}]")
        
        return best_scale_factors
    
    def _compute_ece(self, logits, labels, n_bins=20):
        """
        计算Expected Calibration Error (ECE)
        
        Args:
            logits (torch.Tensor): 模型输出的logits
            labels (torch.Tensor): 真实标签
            n_bins (int): bin的数量
        
        Returns:
            torch.Tensor: ECE值
        """
        softmax_probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmax_probs, dim=1)
        accuracies = predictions.eq(labels)
        
        # 创建bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=self.device)
        ece = torch.tensor(0.0, device=self.device)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # 找到在当前bin中的样本
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def update_scaling_factors(self, epoch_idx, drw_epoch, n_bins=20, 
                              min_scale_factor=0.1, max_iter=200, lr=0.01):
        """
        在训练过程中更新缩放因子
        
        Args:
            epoch_idx (int): 当前epoch
            drw_epoch (int): DRW切换的epoch
            n_bins (int): ECE计算的bin数量
            min_scale_factor (float): 最小缩放因子
            max_iter (int): 最大迭代次数
            lr (float): 学习率
        """
        if epoch_idx < drw_epoch:
            # 早期阶段：使用logits模式
            self.class_scale_factors_logits = self.fit_scaling_factors(
                n_bins=n_bins,
                min_scale_factor=min_scale_factor,
                max_iter=max_iter,
                lr=lr,
                use_logits=True
            )
        else:
            # 后期阶段：使用softmax模式
            self.class_scale_factors_softmax = self.fit_scaling_factors(
                n_bins=n_bins,
                min_scale_factor=min_scale_factor,
                max_iter=max_iter,
                lr=lr,
                use_logits=False
            )
    
    def get_kappa_multi(self):
        """获取当前的kappa_multi参数（logits模式的缩放因子）"""
        return self.class_scale_factors_logits
    
    def get_kappa_add(self):
        """获取当前的kappa_add参数（softmax模式的缩放因子）"""
        return self.class_scale_factors_softmax
    
    def state_dict(self):
        """保存状态"""
        return {
            'class_scale_factors_logits': self.class_scale_factors_logits,
            'class_scale_factors_softmax': self.class_scale_factors_softmax,
        }
    
    def load_state_dict(self, state_dict):
        """加载状态"""
        if 'class_scale_factors_logits' in state_dict:
            self.class_scale_factors_logits = state_dict['class_scale_factors_logits'].to(self.device)
        if 'class_scale_factors_softmax' in state_dict:
            self.class_scale_factors_softmax = state_dict['class_scale_factors_softmax'].to(self.device)
