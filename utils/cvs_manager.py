"""
CVS Manager - 管理CVS损失函数的动态参数计算
"""
import torch
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
    
    def fit_scaling_factors(self, n_bins=20, min_scale_factor=0.1, max_scale_factor=1.0):
        """
        拟合类别缩放因子，使用最小二乘法拟合Confidence-Accuracy斜率
        
        Args:
            n_bins (int): bin的数量
            min_scale_factor (float): 最小缩放因子
            max_scale_factor (float): 最大缩放因子
        
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
        
        # 计算概率和预测
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        scale_factors = torch.ones(self.num_classes, device=self.device)
        
        print(f"[CVSManager] 开始计算缩放因子 (Least Squares Slope Fitting)...")
        
        for c in range(self.num_classes):
            # 找到预测为类别 c 的样本
            # 注意：这里我们关注的是模型对类别 c 的预测置信度与实际准确率的关系
            mask = preds == c
            
            # 如果样本太少，保持默认值 1.0
            if mask.sum() < 10:
                continue
                
            class_conf = probs[mask, c]
            class_acc = (labels[mask] == c).float()
            
            # 分 bin 计算 Avg Conf 和 Avg Acc
            bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=self.device)
            bin_confs = []
            bin_accs = []
            bin_weights = []
            
            for i in range(n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (class_conf > bin_lower) & (class_conf <= bin_upper)
                if in_bin.sum() > 0:
                    avg_conf = class_conf[in_bin].mean()
                    avg_acc = class_acc[in_bin].mean()
                    weight = in_bin.sum().float()
                    
                    bin_confs.append(avg_conf)
                    bin_accs.append(avg_acc)
                    bin_weights.append(weight)
            
            if len(bin_confs) < 2:
                continue
                
            # 转换为 tensor
            bin_confs = torch.stack(bin_confs)
            bin_accs = torch.stack(bin_accs)
            bin_weights = torch.stack(bin_weights)
            
            # 加权最小二乘法拟合 Acc = beta * Conf
            # 目标是最小化 sum(w * (Acc - beta * Conf)^2)
            # 解析解: beta = sum(w * Conf * Acc) / sum(w * Conf^2)
            numerator = torch.sum(bin_weights * bin_confs * bin_accs)
            denominator = torch.sum(bin_weights * bin_confs ** 2)
            
            if denominator > 1e-6:
                beta = numerator / denominator
                # kappa = 1 / beta
                # 如果 beta < 1 (Overconfident), kappa > 1, 降低 confidence
                # 如果 beta > 1 (Underconfident), kappa < 1, 提高 confidence
                kappa = 1.0 / (beta + 1e-6)
                
                # 限制范围
                # 限制最大缩放因子，防止过度调整
                kappa = torch.clamp(kappa, min=min_scale_factor, max=max_scale_factor)
                scale_factors[c] = kappa
        
        print(f"[CVSManager] 拟合完成")
        print(f"[CVSManager] 缩放因子范围: [{scale_factors.min().item():.4f}, {scale_factors.max().item():.4f}]")
        
        return scale_factors
    
    
    def update_scaling_factors(self, epoch_idx, drw_epoch, n_bins=20, 
                              min_scale_factor=0.1, max_scale_factor=1.0, momentum=0.9):
        """
        在训练过程中更新缩放因子
        
        Args:
            epoch_idx (int): 当前epoch
            drw_epoch (int): DRW切换的epoch
            n_bins (int): ECE计算的bin数量
            min_scale_factor (float): 最小缩放因子
            max_scale_factor (float): 最大缩放因子
            momentum (float): 动量系数，用于平滑更新
        """
        if epoch_idx < drw_epoch:
            # 早期阶段：使用logits模式
            new_factors = self.fit_scaling_factors(
                n_bins=n_bins,
                min_scale_factor=min_scale_factor,
                max_scale_factor=max_scale_factor
            )
            # 使用动量更新平滑参数变化
            self.class_scale_factors_logits = momentum * self.class_scale_factors_logits + (1 - momentum) * new_factors
        else:
            # 后期阶段：使用softmax模式
            new_factors = self.fit_scaling_factors(
                n_bins=n_bins,
                min_scale_factor=min_scale_factor,
                max_scale_factor=max_scale_factor
            )
            self.class_scale_factors_softmax = momentum * self.class_scale_factors_softmax + (1 - momentum) * new_factors
    
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
