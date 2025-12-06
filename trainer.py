import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224

import datasets
from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator
from utils.templates import ZEROSHOT_TEMPLATES


def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(backbone_name, prec):
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=True).eval()

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model


class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None
        
        # 初始化类别memory bank
        samples_per_class = getattr(cfg, 'memory_bank_size', 1000)  # 默认每个类别1000个样本
        self._init_class_memory_bank(samples_per_class)
        
        # 初始化类别缩放因子（用于CVS损失函数的动态调整）
        # 两种模式的缩放因子：logits模式用于kappa_multi，softmax模式用于kappa_add
        self.class_scale_factors_softmax = torch.ones(self.num_classes, device=self.device)  # fit_use_logits=False，替换kappa_add
        self.class_scale_factors_logits = torch.ones(self.num_classes, device=self.device)   # fit_use_logits=True，替换kappa_multi
        
        # 初始化时间统计
        self.time_stats = {
            'forward_time': 0.0,
            'backward_time': 0.0,
            'fit_time': 0.0,
            'forward_count': 0,
            'backward_count': 0,
            'fit_count': 0,
            'epoch_times': []
        }
        
        # 初始化时间日志
        self._init_time_logger()

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if cfg.tte:
            if cfg.tte_mode == "fivecrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.FiveCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "tencrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.TenCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "randaug":
                _resize_and_flip = transforms.Compose([
                    transforms.RandomResizedCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.Lambda(lambda image: torch.stack([_resize_and_flip(image) for _ in range(cfg.randaug_times)])),
                    transforms.Normalize(mean, std),
                ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])

        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.train_loader = DataLoader(train_dataset,
            batch_size=cfg.micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=64, sampler=init_sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
            batch_size=64, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Total training points:", sum(self.cls_num_list))
        # print(self.cls_num_list)

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                print("No initialization with head")
            
            torch.cuda.empty_cache()
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                      {"params": self.head.parameters()}],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = WeightedCELoss(cls_num_list=self.cls_num_list)
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM": # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB": # https://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW": # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS": # https://arxiv.org/abs/2007.10740
            self.criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list, tau=cfg.la_tau)
        elif cfg.loss_type == "LADE": # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "CVS":
            self.criterion = CVSLoss(cls_num_list=cls_num_list, gamma=cfg.cvs_gamma, tau=cfg.cvs_tau)
        elif cfg.loss_type == "VS":
            self.criterion = VSLoss(cls_num_list=cls_num_list, gamma=cfg.vs_gamma, tau=cfg.vs_tau)
        
    def get_tokenized_prompts(self, classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        if cfg.prompt == "ensemble":
            all_text_features = []
            for template in tqdm(ZEROSHOT_TEMPLATES['imagenet']):
                prompts = self.get_tokenized_prompts(classnames, template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_text_features.append(text_features)
            all_text_features = torch.stack(all_text_features)
            text_features = all_text_features.mean(dim=0)
        elif cfg.prompt == "descriptor":
            with open("utils/descriptors_imagenet.json") as f:
                descriptors = json.load(f)
            template = "{}"
            all_class_features = []
            for cn in tqdm(classnames):
                prompts = self.get_tokenized_prompts(descriptors[cn], template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_class_features.append(text_features.mean(dim=0))
            text_features = torch.stack(all_class_features)
        elif cfg.prompt == "classname":
            template = "{}"
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        elif cfg.prompt == "default":
            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)

    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features, all_labels = [], []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head.apply_weight(class_means)

    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features, all_labels = [], []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    def _init_time_logger(self):
        """初始化时间统计日志器"""
        # 创建时间日志目录
        log_dir = os.path.join(self.cfg.output_dir, "timing_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置时间日志器
        self.time_logger = logging.getLogger('timing_logger')
        self.time_logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        for handler in self.time_logger.handlers[:]:
            self.time_logger.removeHandler(handler)
        
        # 创建文件处理器
        log_file = os.path.join(log_dir, f"timing_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.time_logger.addHandler(file_handler)
        self.time_logger.addHandler(console_handler)
        
        # 防止日志传播到根日志器
        self.time_logger.propagate = False
        
        self.time_logger.info("时间统计日志器初始化完成")

    def _log_time_stats(self, phase, duration, extra_info=None):
        """记录时间统计
        
        Args:
            phase (str): 阶段名称 ('forward', 'backward', 'fit', 'epoch')
            duration (float): 持续时间（秒）
            extra_info (dict): 额外信息
        """
        # 更新统计
        if phase in ['forward', 'backward', 'fit']:
            self.time_stats[f'{phase}_time'] += duration
            self.time_stats[f'{phase}_count'] += 1
        elif phase == 'epoch':
            self.time_stats['epoch_times'].append(duration)
        
        # 记录日志
        info_str = f"{phase.upper()}: {duration:.4f}s"
        if extra_info:
            for key, value in extra_info.items():
                info_str += f", {key}: {value}"
        
        self.time_logger.info(info_str)
        
        # 写入tensorboard
        if hasattr(self, '_writer') and self._writer is not None:
            step = getattr(self, '_current_step', 0)
            self._writer.add_scalar(f"timing/{phase}_time", duration, step)
            if phase in ['forward', 'backward', 'fit']:
                avg_time = self.time_stats[f'{phase}_time'] / max(1, self.time_stats[f'{phase}_count'])
                self._writer.add_scalar(f"timing/{phase}_avg_time", avg_time, step)

    def _print_time_summary(self):
        """打印时间统计摘要"""
        forward_avg = self.time_stats['forward_time'] / max(1, self.time_stats['forward_count'])
        backward_avg = self.time_stats['backward_time'] / max(1, self.time_stats['backward_count'])
        fit_avg = self.time_stats['fit_time'] / max(1, self.time_stats['fit_count'])
        
        total_epoch_time = sum(self.time_stats['epoch_times'])
        avg_epoch_time = np.mean(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else 0
        
        summary = f"""
========== 训练时间统计摘要 ==========
前向传播:
  - 总次数: {self.time_stats['forward_count']}
  - 总时间: {self.time_stats['forward_time']:.2f}s
  - 平均时间: {forward_avg:.4f}s

反向传播:
  - 总次数: {self.time_stats['backward_count']}
  - 总时间: {self.time_stats['backward_time']:.2f}s
  - 平均时间: {backward_avg:.4f}s

缩放因子拟合:
  - 总次数: {self.time_stats['fit_count']}
  - 总时间: {self.time_stats['fit_time']:.2f}s
  - 平均时间: {fit_avg:.2f}s

Epoch统计:
  - 总epoch数: {len(self.time_stats['epoch_times'])}
  - 总epoch时间: {total_epoch_time:.2f}s
  - 平均epoch时间: {avg_epoch_time:.2f}s

总训练时间: {self.time_stats['forward_time'] + self.time_stats['backward_time'] + self.time_stats['fit_time']:.2f}s
=====================================
        """
        
        self.time_logger.info(summary)
        print(summary)

    def _init_class_memory_bank(self, samples_per_class=1000):
        """
        初始化类别memory bank，为每个类别维护一个固定大小的FIFO队列
        
        Args:
            samples_per_class (int): 每个类别最多保存的样本数量
        """
        self.samples_per_class = samples_per_class
        self.class_memory_bank = {}
        
        # 为每个类别初始化空的memory bank
        for class_idx in range(self.num_classes):
            # 对于样本数少于预设长度的类别，调整bank大小为实际样本数
            class_sample_count = self.cls_num_list[class_idx]
            actual_bank_size = min(samples_per_class, class_sample_count)
            
            self.class_memory_bank[class_idx] = {
                'logits': [],  # 存储logits
                'labels': [],  # 存储labels (用于验证)
                'count': 0,    # 当前存储的样本数
                'pointer': 0,  # 循环写入的指针
                'max_size': actual_bank_size,  # 该类别的实际bank大小
                'is_full': False  # 标记该类别是否已收集满
            }
        
        # 统计调整情况
        adjusted_classes = sum(1 for i in range(self.num_classes) 
                              if self.cls_num_list[i] < samples_per_class)
        
        print(f"初始化类别memory bank，默认每个类别最多保存 {samples_per_class} 个样本")
        if adjusted_classes > 0:
            print(f"  其中 {adjusted_classes} 个少样本类别调整为实际样本数量")
            # 显示前几个调整的类别作为示例
            examples = []
            for class_idx in range(min(5, self.num_classes)):
                if self.cls_num_list[class_idx] < samples_per_class:
                    examples.append(f"类别{class_idx}:{self.cls_num_list[class_idx]}样本")
            if examples:
                print(f"  示例: {', '.join(examples)}")
                if adjusted_classes > 5:
                    print(f"  ... 还有 {adjusted_classes - 5} 个类别被调整")
    
    def _update_class_memory_bank(self, logits, labels):
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
            # 找到属于当前类别的样本
            mask = (labels_cpu == class_idx)
            if mask.sum() == 0:
                continue
            
            class_logits = logits_cpu[mask]
            class_labels = labels_cpu[mask]
            
            bank = self.class_memory_bank[class_idx]
            max_size = bank['max_size']
            
            # 如果bank还没满，直接添加
            if bank['count'] < max_size:
                bank['logits'].extend(class_logits)
                bank['labels'].extend(class_labels)
                bank['count'] += len(class_logits)
                
                # 检查是否已满
                if bank['count'] >= max_size:
                    bank['is_full'] = True
                    # 对于少样本类别，截断到精确大小
                    if max_size < self.samples_per_class and bank['count'] > max_size:
                        bank['logits'] = bank['logits'][:max_size]
                        bank['labels'] = bank['labels'][:max_size]
                        bank['count'] = max_size
            else:
                # bank已满，使用FIFO策略替换（只对大样本类别）
                if max_size == self.samples_per_class:
                    for i, (logit, label) in enumerate(zip(class_logits, class_labels)):
                        # 循环写入
                        idx = bank['pointer'] % max_size
                        bank['logits'][idx] = logit
                        bank['labels'][idx] = label
                        bank['pointer'] = (bank['pointer'] + 1) % max_size
    
    def _get_memory_bank_data(self):
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
                # 只取有效的数据
                valid_count = min(bank['count'], len(bank['logits']))
                if valid_count > 0:
                    class_logits = torch.stack(bank['logits'][:valid_count])
                    class_labels = torch.stack(bank['labels'][:valid_count])
                    all_logits.append(class_logits)
                    all_labels.append(class_labels)
        
        if len(all_logits) == 0:
            return None, None
        
        # 合并所有类别的数据
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return logits, labels
    
    def _print_memory_bank_stats(self):
        """打印memory bank的统计信息"""
        total_samples = 0
        non_empty_classes = 0
        full_classes = 0
        adjusted_classes = 0
        
        print("Memory Bank 统计信息:")
        for class_idx in range(min(10, self.num_classes)):  # 只显示前10个类
            bank = self.class_memory_bank[class_idx]
            if bank['count'] > 0:
                non_empty_classes += 1
                total_samples += bank['count']
                
                if bank['is_full']:
                    full_classes += 1
                
                if bank['max_size'] < self.samples_per_class:
                    adjusted_classes += 1
                
                status = "已满" if bank['is_full'] else "未满"
                if bank['max_size'] < self.samples_per_class:
                    status += f"(调整为{bank['max_size']})"
                
                print(f"  类别 {class_idx}: {bank['count']}/{bank['max_size']} 样本 [{status}]")
        
        if self.num_classes > 10:
            # 统计剩余类别
            for class_idx in range(10, self.num_classes):
                bank = self.class_memory_bank[class_idx]
                if bank['count'] > 0:
                    non_empty_classes += 1
                    total_samples += bank['count']
                    if bank['is_full']:
                        full_classes += 1
                    if bank['max_size'] < self.samples_per_class:
                        adjusted_classes += 1
            print(f"  ... (还有 {non_empty_classes - min(10, non_empty_classes)} 个非空类别)")
        
        print(f"  总样本数: {total_samples}, 非空类别数: {non_empty_classes}/{self.num_classes}")
        print(f"  已满类别数: {full_classes}, 调整大小类别数: {adjusted_classes}")
        if non_empty_classes > 0:
            print(f"  平均每类样本数: {total_samples / non_empty_classes:.1f}")

    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            epoch_start_time = time.time()
            self.tuner.train()
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)

                # 设置当前步数用于tensorboard
                self._current_step = epoch_idx * num_batches + batch_idx

                if cfg.prec == "amp":
                    with autocast():
                        # 记录前向传播时间
                        forward_start = time.time()
                        output = self.model(image)
                        forward_time = time.time() - forward_start
                        self._log_time_stats('forward', forward_time, {
                            'epoch': epoch_idx + 1,
                            'batch': batch_idx + 1,
                            'batch_size': image.size(0)
                        })
                        
                        # 更新类别memory bank
                        with torch.no_grad():
                            self._update_class_memory_bank(output, label)
                        
                        if cfg.loss_type == 'CVS':
                            if epoch_idx < cfg.drw_epoch:
                                # 使用logits模式的缩放因子替换cfg.cvs_kappa_multi
                                loss = self.criterion(output, label, 0, self.class_scale_factors_logits, 0)
                            else:
                                # 使用softmax模式的缩放因子替换cfg.cvs_kappa_add
                                loss = self.criterion(output, label, cfg.cvs_tro, 0, self.class_scale_factors_softmax)
                        elif cfg.loss_type == 'LDAM':
                            if epoch_idx < cfg.drw_epoch:
                                loss = self.criterion(output, label, None)
                            else:
                                loss = self.criterion(output, label, cfg.ldam_tro)
                        elif cfg.loss_type == 'CE':
                            if epoch_idx < cfg.drw_epoch:
                                loss = self.criterion(output, label, None)
                            else:
                                loss = self.criterion(output, label, cfg.ce_tro)
                        elif cfg.loss_type == 'VS':
                            if epoch_idx < cfg.drw_epoch:
                                loss = self.criterion(output, label, None)
                            else:
                                loss = self.criterion(output, label, cfg.vs_tro)
                        else:
                            loss = self.criterion(output, label)
                        loss_micro = loss / self.accum_step
                        
                        # 记录反向传播时间
                        backward_start = time.time()
                        self.scaler.scale(loss_micro).backward()
                        backward_time = time.time() - backward_start
                        self._log_time_stats('backward', backward_time, {
                            'epoch': epoch_idx + 1,
                            'batch': batch_idx + 1,
                            'loss': loss.item()
                        })
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    # 记录前向传播时间
                    forward_start = time.time()
                    output = self.model(image)
                    forward_time = time.time() - forward_start
                    self._log_time_stats('forward', forward_time, {
                        'epoch': epoch_idx + 1,
                        'batch': batch_idx + 1,
                        'batch_size': image.size(0)
                    })
                    
                    # 更新类别memory bank
                    with torch.no_grad():
                        self._update_class_memory_bank(output, label)
                    
                    if cfg.loss_type == 'CVS':
                        if epoch_idx < cfg.drw_epoch:
                            # 使用logits模式的缩放因子替换cfg.cvs_kappa_multi
                            loss = self.criterion(output, label, 0, self.class_scale_factors_logits, 0)
                        else:
                            # 使用softmax模式的缩放因子替换cfg.cvs_kappa_add
                            loss = self.criterion(output, label, cfg.cvs_tro, 0, self.class_scale_factors_softmax)
                    elif cfg.loss_type == 'LDAM':
                        if epoch_idx < cfg.drw_epoch:
                            loss = self.criterion(output, label, None)
                        else:
                            loss = self.criterion(output, label, cfg.ldam_tro)
                    elif cfg.loss_type == 'CE':
                        if epoch_idx < cfg.drw_epoch:
                            loss = self.criterion(output, label, None)
                        else:
                            loss = self.criterion(output, label, cfg.ce_tro)
                    elif cfg.loss_type == 'VS':
                        if epoch_idx < cfg.drw_epoch:
                            loss = self.criterion(output, label, None)
                        else:
                            loss = self.criterion(output, label, cfg.vs_tro)
                    else:
                        loss = self.criterion(output, label)
                    loss_micro = loss / self.accum_step
                    
                    # 记录反向传播时间
                    backward_start = time.time()
                    loss_micro.backward()
                    backward_time = time.time() - backward_start
                    self._log_time_stats('backward', backward_time, {
                        'epoch': epoch_idx + 1,
                        'batch': batch_idx + 1,
                        'loss': loss.item()
                    })
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_accs))
                many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
                end = time.time()

            self.sched.step()
            torch.cuda.empty_cache()
            
            # 每个epoch结束时更新CVS缩放因子（如果使用CVS损失）
            if cfg.loss_type == 'CVS':
                fit_start_time = time.time()
                self.update_cvs_scaling_factors(epoch_idx)
                fit_time = time.time() - fit_start_time
                self._log_time_stats('fit', fit_time, {
                    'epoch': epoch_idx + 1,
                    'phase': 'early' if epoch_idx < cfg.drw_epoch else 'late'
                })
            
            # 记录epoch时间
            epoch_time = time.time() - epoch_start_time
            self._log_time_stats('epoch', epoch_time, {
                'epoch': epoch_idx + 1,
                'batches': num_batches
            })
            
            # Save checkpoint at specified frequency
            if cfg.save_freq > 0 and (epoch_idx + 1) % cfg.save_freq == 0:
                self.save_model(cfg.output_dir, epoch_idx + 1)

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # 拟合类别缩放因子（使用memory bank中的数据）
        print("\n" + "="*60)
        print("训练结束后最终拟合类别缩放因子")
        self._print_memory_bank_stats()
        print("="*60)
        
        # 最终拟合两种模式的缩放因子
        print("执行最终的双模式缩放因子拟合...")
        
        # 拟合logits模式的缩放因子（替换kappa_multi）
        final_fit_start = time.time()
        self.class_scale_factors_logits = self.fit_class_scaling_factors(
            n_bins=20, 
            min_scale_factor=getattr(cfg, 'cvs_kappa_multi', 0.1), 
            max_iter=1000, 
            lr=0.01, 
            use_logits=True
        )
        logits_fit_time = time.time() - final_fit_start
        self._log_time_stats('fit', logits_fit_time, {
            'phase': 'final_logits',
            'mode': 'logits',
            'max_iter': 1000
        })
        
        # 拟合softmax模式的缩放因子（替换kappa_add）
        softmax_fit_start = time.time()
        self.class_scale_factors_softmax = self.fit_class_scaling_factors(
            n_bins=20, 
            min_scale_factor=getattr(cfg, 'cvs_kappa_add', 0.1), 
            max_iter=1000, 
            lr=0.01, 
            use_logits=False
        )
        softmax_fit_time = time.time() - softmax_fit_start
        self._log_time_stats('fit', softmax_fit_time, {
            'phase': 'final_softmax',
            'mode': 'softmax',
            'max_iter': 1000
        })
        
        print("最终类别缩放因子拟合完成")
        print(f"  Logits模式范围(替换kappa_multi): [{torch.min(self.class_scale_factors_logits):.4f}, {torch.max(self.class_scale_factors_logits):.4f}]")
        print(f"  Softmax模式范围(替换kappa_add): [{torch.min(self.class_scale_factors_softmax):.4f}, {torch.max(self.class_scale_factors_softmax):.4f}]")

        # 打印时间统计摘要
        self._print_time_summary()

        # save model
        self.save_model(cfg.output_dir)

        self.test()

        # Close writer
        self._writer.close()
        
        # 关闭时间日志器
        for handler in self.time_logger.handlers[:]:
            handler.close()
            self.time_logger.removeHandler(handler)

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            if _ncrops <= 5:
                output = self.model(image)
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    output.append(self.model(image[:, k]))
                output = torch.stack(output).mean(dim=0)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]

    def save_model(self, directory, epoch=None):
        """Save model checkpoint
        
        Args:
            directory: Directory path or full file path to save the checkpoint
            epoch: Optional epoch number to include in checkpoint and filename
        """
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
            "head": head_dict
        }
        
        # Add epoch to checkpoint if provided
        if epoch is not None:
            checkpoint["epoch"] = epoch
            
        # Add class scaling factors if available
        if hasattr(self, 'class_scale_factors_softmax') and hasattr(self, 'class_scale_factors_logits'):
            checkpoint["class_scale_factors_softmax"] = self.class_scale_factors_softmax
            checkpoint["class_scale_factors_logits"] = self.class_scale_factors_logits
            print(f"保存CVS双模式缩放因子到checkpoint")
        elif hasattr(self, 'class_scale_factors'):
            checkpoint["class_scale_factors"] = self.class_scale_factors
            print(f"保存类别缩放因子到checkpoint")

        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # Determine save path
        if epoch is not None and os.path.isdir(directory):
            # For epoch-specific saves with directory
            save_path = os.path.join(directory, f"checkpoint_epoch_{epoch}.pth")
        elif os.path.isdir(directory):
            # For final model save with directory
            save_path = os.path.join(directory, "checkpoint.pth.tar")
        else:
            # Direct file path provided
            save_path = directory

        # Save checkpoint
        torch.save(checkpoint, save_path)
        
        if epoch is not None:
            print(f"Epoch {epoch} checkpoint saved to {save_path}")
        else:
            print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        # Try to load specific checkpoint file if directory is a file path
        if os.path.isfile(directory):
            load_path = directory
        else:
            # Default behavior: load from directory
            load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]

        print("Loading weights from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict, strict=False)

        if head_dict["weight"].shape == self.head.weight.shape:
            self.head.load_state_dict(head_dict, strict=False)
            
        # Load class scaling factors if available
        if "class_scale_factors_softmax" in checkpoint and "class_scale_factors_logits" in checkpoint:
            self.class_scale_factors_softmax = checkpoint["class_scale_factors_softmax"]
            self.class_scale_factors_logits = checkpoint["class_scale_factors_logits"]
            print(f"加载CVS双模式缩放因子")
            print(f"  Softmax模式形状: {self.class_scale_factors_softmax.shape}")
            print(f"  Logits模式形状: {self.class_scale_factors_logits.shape}")
        elif "class_scale_factors" in checkpoint:
            # 兼容旧版本的单一缩放因子
            self.class_scale_factors = checkpoint["class_scale_factors"]
            print(f"加载类别缩放因子，形状: {self.class_scale_factors.shape}")
            # 将单一缩放因子复制到两个新的属性中
            self.class_scale_factors_softmax = self.class_scale_factors.clone()
            self.class_scale_factors_logits = self.class_scale_factors.clone()
        else:
            print("checkpoint中未找到类别缩放因子")

    def fit_class_scaling_factors(self, n_bins=20, min_scale_factor=0.1, max_iter=1000, lr=0.01, use_logits=True):
        """
        对每个类拟合缩放因子 a，使得 accuracy_y = a * confidence_y
        使用ECE分bin的方法，在每个bin内计算平均准确率
        使用类别memory bank中收集的数据
        
        Args:
            n_bins (int): 置信度分bin的数量
            min_scale_factor (float): 最小缩放因子，用于线性缩放
            max_iter (int): 最大迭代次数
            lr (float): 学习率
            use_logits (bool): True=对原始logits拟合, False=对softmax概率拟合
            
        Returns:
            torch.Tensor: 每个类的缩放因子，形状为 (num_classes,)
        """
        # 从memory bank获取数据
        logits, labels = self._get_memory_bank_data()
        
        if logits is None or labels is None:
            print("警告：memory bank中没有数据，无法进行缩放因子拟合")
            return torch.ones(self.num_classes, device=self.device)
        
        # 移动到GPU
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        
        num_classes, num_samples = logits.shape[1], logits.shape[0]
        
        # 根据模式选择拟合目标
        if use_logits:
            print(f"  -> 使用原始logits模式拟合缩放因子 (min_scale={min_scale_factor})")
            # 使用原始logits中对应类别的值作为置信度
            target_logits = logits
            confidences = torch.gather(target_logits, 1, labels.unsqueeze(1)).squeeze(1)
            # 对logits进行softmax得到预测
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        else:
            print(f"  -> 使用softmax概率模式拟合缩放因子 (min_scale={min_scale_factor})")
            # 计算softmax概率和置信度
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]  # 最大概率作为置信度
        
        accuracies = (predictions == labels).float()
        
        print(f"开始拟合类别缩放因子，总类别数: {num_classes}, 总样本数: {num_samples}, 分bin数: {n_bins}")
        print(f"拟合模式: {'原始logits' if use_logits else 'softmax概率'}")
        
        # 为每个类别计算bin数据
        class_bin_data = {}
        
        if use_logits:
            # 对于logits模式，每个类别独立计算bin边界
            for class_idx in range(num_classes):
                mask = (labels == class_idx)
                if mask.sum() == 0:
                    continue
                    
                class_confidences = confidences[mask]
                class_accuracies = accuracies[mask]
                
                # 为当前类别计算bin边界（基于该类别的logits分布）
                min_conf = class_confidences.min().item()
                max_conf = class_confidences.max().item()
                
                if max_conf <= min_conf:  # 避免除零
                    continue
                    
                bin_boundaries = torch.linspace(min_conf, max_conf, n_bins + 1, device=self.device)
                
                bin_confidences, bin_accuracies, bin_counts = [], [], []
                
                # 对每个bin计算平均置信度和准确率
                for bin_idx in range(n_bins):
                    bin_lower = bin_boundaries[bin_idx]
                    bin_upper = bin_boundaries[bin_idx + 1]
                    
                    # 最后一个bin包含右边界
                    if bin_idx == n_bins - 1:
                        bin_mask = (class_confidences >= bin_lower) & (class_confidences <= bin_upper)
                    else:
                        bin_mask = (class_confidences >= bin_lower) & (class_confidences < bin_upper)
                    
                    if bin_mask.sum() > 0:
                        bin_conf = class_confidences[bin_mask].mean()
                        bin_acc = class_accuracies[bin_mask].mean()
                        bin_count = bin_mask.sum().item()
                        
                        bin_confidences.append(bin_conf.detach())
                        bin_accuracies.append(bin_acc.detach())
                        bin_counts.append(bin_count)
                
                if len(bin_confidences) > 0:
                    class_bin_data[class_idx] = {
                        'confidences': torch.stack(bin_confidences),
                        'accuracies': torch.stack(bin_accuracies),
                        'counts': bin_counts,
                        'total_count': mask.sum().item()
                    }
        else:
            # 对于概率模式，使用全局bin边界[0,1]
            bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=self.device)
            
            for class_idx in range(num_classes):
                mask = (labels == class_idx)
                if mask.sum() == 0:
                    continue
                    
                class_confidences = confidences[mask]
                class_accuracies = accuracies[mask]
                
                bin_confidences, bin_accuracies, bin_counts = [], [], []
                
                # 对每个bin计算平均置信度和准确率
                for bin_idx in range(n_bins):
                    bin_lower = bin_boundaries[bin_idx]
                    bin_upper = bin_boundaries[bin_idx + 1]
                    
                    # 最后一个bin包含右边界
                    if bin_idx == n_bins - 1:
                        bin_mask = (class_confidences >= bin_lower) & (class_confidences <= bin_upper)
                    else:
                        bin_mask = (class_confidences >= bin_lower) & (class_confidences < bin_upper)
                    
                    if bin_mask.sum() > 0:
                        bin_conf = class_confidences[bin_mask].mean()
                        bin_acc = class_accuracies[bin_mask].mean()
                        bin_count = bin_mask.sum().item()
                        
                        bin_confidences.append(bin_conf.detach())
                        bin_accuracies.append(bin_acc.detach())
                        bin_counts.append(bin_count)
                
                if len(bin_confidences) > 0:
                    class_bin_data[class_idx] = {
                        'confidences': torch.stack(bin_confidences),
                        'accuracies': torch.stack(bin_accuracies),
                        'counts': bin_counts,
                        'total_count': mask.sum().item()
                    }
        
        if len(class_bin_data) == 0:
            print("没有有效的类别数据，无法进行拟合")
            return torch.ones(num_classes, device=self.device)
        
        print(f"有效类别数: {len(class_bin_data)}")
        
        # 为所有类别并行拟合缩放因子和偏置项
        scale_factors = torch.ones(num_classes, device=self.device, requires_grad=True)
        bias_factors = torch.zeros(num_classes, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([scale_factors, bias_factors], lr=lr)
        
        # 优化循环
        best_loss = float('inf')
        patience = 100
        no_improve_count = 0
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            total_loss = None
            
            # 对每个有数据的类别计算损失（并行处理）
            for class_idx, data in class_bin_data.items():
                # 获取这个类别的缩放因子和偏置项
                scale = scale_factors[class_idx]
                bias = bias_factors[class_idx]
                
                # 预测准确率: accuracy = scale * confidence + bias
                predicted_accs = scale * data['confidences'] + bias
                
                # 计算MSE损失
                mse_loss = torch.mean((predicted_accs - data['accuracies']) ** 2)
                
                if total_loss is None:
                    total_loss = mse_loss
                else:
                    total_loss = total_loss + mse_loss
            
            # 添加正则化
            # 只对有数据的类别进行正则化
            valid_class_indices = torch.tensor(list(class_bin_data.keys()), device=self.device)
            scale_reg = torch.mean((scale_factors[valid_class_indices] - 1.0) ** 2)
            bias_reg = torch.mean(bias_factors[valid_class_indices] ** 2)
            reg_loss = 0.001 * scale_reg + 0.001 * bias_reg
            
            if total_loss is not None:
                total_loss = total_loss + reg_loss
            else:
                total_loss = reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            # 限制参数在合理范围内
            with torch.no_grad():
                scale_factors.clamp_(0.01, 10.0)
                bias_factors.clamp_(-1.0, 1.0)
            
            # 早停检查
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience:
                print(f"早停于迭代 {iteration}, 损失: {best_loss:.6f}")
                break
                
            # 打印进度
            if iteration % 100 == 0:
                main_loss = total_loss.item() - reg_loss.item()
                print(f"迭代 {iteration}: 总损失 {total_loss.item():.6f}, "
                      f"拟合损失 {main_loss:.6f}, 正则化损失 {reg_loss.item():.6f}")
                # 显示前几个类别的参数
                for i, class_idx in enumerate(sorted(list(class_bin_data.keys())[:3])):
                    print(f"    类别{class_idx}: scale={scale_factors[class_idx].item():.4f}, bias={bias_factors[class_idx].item():.4f}")
        
        # 对所有缩放因子进行线性缩放（只对有数据的类别）
        with torch.no_grad():
            valid_scale_factors = scale_factors[valid_class_indices]
            max_factor = torch.max(valid_scale_factors)
            min_factor = torch.min(valid_scale_factors)
            
            # 线性缩放公式: new_value = min_scale + (old_value - min_old) * (1 - min_scale) / (max_old - min_old)
            if max_factor > min_factor:
                scale_factors[valid_class_indices] = min_scale_factor + (valid_scale_factors - min_factor) * (1.0 - min_scale_factor) / (max_factor - min_factor)
            else:
                scale_factors[valid_class_indices].fill_(1.0)  # 如果所有缩放因子相同，设为1
        
        print(f"拟合完成。缩放因子范围: [{torch.min(scale_factors):.4f}, {torch.max(scale_factors):.4f}]")
        print(f"拟合完成。偏置项范围: [{torch.min(bias_factors):.4f}, {torch.max(bias_factors):.4f}]")
        
        # 打印每个类的信息
        print("各类别缩放因子和偏置项统计:")
        for class_idx in sorted(list(class_bin_data.keys())[:10]):  # 只打印前10个类
            data = class_bin_data[class_idx]
            print(f"  类别 {class_idx}: 缩放因子 {scale_factors[class_idx]:.4f}, 偏置项 {bias_factors[class_idx]:.4f}, "
                  f"总样本数 {data['total_count']}, 有效bin数 {len(data['counts'])}")
        
        if len(class_bin_data) > 10:
            print(f"  ... (还有 {len(class_bin_data) - 10} 个类别)")
        
        return scale_factors.detach()
    
    def update_cvs_scaling_factors(self, epoch_idx):
        """
        更新CVS损失函数的缩放因子
        在每个epoch结束时调用，使用memory bank中的数据拟合两种模式的缩放因子
        根据当前epoch阶段只计算需要的缩放因子
        
        Args:
            epoch_idx (int): 当前epoch索引
        """
        # 检查是否有足够的数据进行拟合
        logits, labels = self._get_memory_bank_data()
        if logits is None or labels is None:
            print(f"Epoch {epoch_idx + 1}: Memory bank中数据不足，跳过缩放因子更新")
            return
        
        print(f"\nEpoch {epoch_idx + 1}: 更新CVS缩放因子")
        
        if epoch_idx < self.cfg.drw_epoch:
            # 早期训练阶段：只拟合logits模式的缩放因子（用于kappa_multi参数）
            print(f"  当前为早期训练阶段 (epoch {epoch_idx + 1} < drw_epoch {self.cfg.drw_epoch})，仅更新logits模式缩放因子")
            logits_start = time.time()
            self.class_scale_factors_logits = self.fit_class_scaling_factors(
                n_bins=20, 
                min_scale_factor=getattr(self.cfg, 'cvs_kappa_multi', 0.1), 
                max_iter=500, 
                lr=0.01, 
                use_logits=True   # logits模式
            )
            logits_time = time.time() - logits_start
            print(f"  Logits模式缩放因子范围: [{torch.min(self.class_scale_factors_logits):.4f}, {torch.max(self.class_scale_factors_logits):.4f}]")
            print(f"  Logits模式拟合耗时: {logits_time:.2f}s")
        else:
            # 后期训练阶段：只拟合softmax模式的缩放因子（用于kappa_add参数）
            print(f"  当前为后期训练阶段 (epoch {epoch_idx + 1} >= drw_epoch {self.cfg.drw_epoch})，仅更新softmax模式缩放因子")
            softmax_start = time.time()
            self.class_scale_factors_softmax = self.fit_class_scaling_factors(
                n_bins=20, 
                min_scale_factor=getattr(self.cfg, 'cvs_kappa_add', 0.1), 
                max_iter=500, 
                lr=0.01, 
                use_logits=False  # softmax模式
            )
            softmax_time = time.time() - softmax_start
            print(f"  Softmax模式缩放因子范围: [{torch.min(self.class_scale_factors_softmax):.4f}, {torch.max(self.class_scale_factors_softmax):.4f}]")
            print(f"  Softmax模式拟合耗时: {softmax_time:.2f}s")

