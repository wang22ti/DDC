# A Unified Generalization Analysis of Re-Weighting and Logit-Adjustment for Imbalanced Learning

This is a Pytorch implementation of our NeurIPS 2023 Spotlight paper: [A Unified Generalization Analysis of Re-Weighting and Logit-Adjustment for Imbalanced Learning](https://arxiv.org/abs/2310.04752).

> **Abstract:** Real-world datasets are typically imbalanced in the sense that only a few classes have numerous samples, while many classes are associated with only a few samples. As a result, a naïve ERM learning process will be biased towards the majority classes, making it difficult to generalize to the minority classes. To address this issue, one simple but effective approach is to modify the loss function to emphasize the learning on minority classes, such as re-weighting the losses or adjusting the logits via class-dependent terms. However, existing generalization analysis of such losses is still coarse-grained and fragmented, failing to explain some empirical results. To bridge this gap, we propose a novel technique named data-dependent contraction to capture how these modified losses handle different classes. On top of this technique, a fine-grained generalization bound is established for imbalanced learning, which helps reveal the mystery of re-weighting and logit-adjustment in a unified manner. Furthermore, a principled learning algorithm is developed based on the theoretical insights. Finally, the empirical results on benchmark datasets not only validate the theoretical results but also demonstrate the effectiveness of the proposed method.

Our codes are based on the repositories [Escaping Saddle Points for Effective Generalization on Class-Imbalanced Data](https://github.com/val-iisc/saddle-longtail).

## Dependencies

Please refer to the `requirements.yml` in the root folder.

## Scripts

Pleas refer to the `bash_script.sh` in the root folder.

## Results

### CIFAR-100

- 400 Epochs with tuned wd, SAM, and RandAug

| | LT ($\rho=100$)  | LT ($\rho=10$)  | Step ($\rho=100$)  | Step ($\rho=10$) |
|  ----  | ----  | ----  | ----  | ----  |
| CE | 45.70 ± 0.35 | 63.11 ± 0.19 |  41.98 ± 0.15 | 59.26 ± 0.24 |
| LDAM + DRW | 50.67 ± 0.25 | 61.55 ± 0.05 | 50.62 ± 0.24 | 60.08 ± 0.14 |
| VS | 52.09 ± 0.12 | 65.23 ± 0.24 | 49.77 ± 0.42 | 64.28 ± 0.04 |
| VS + ADRW + TLA (Ours) | **53.05 ± 0.12** | **65.59 ± 0.28** | **51.69 ± 0.29** | **64.98 ± 0.13** |

<!-- - 200 Epochs w/o SAM and RandAug -->

### CIFAR-10

- 400 Epochs with tuned wd, SAM, and RandAug

| | LT ($\rho=100$)  | LT ($\rho=10$)  | Step ($\rho=100$)  | Step ($\rho=10$) |
|  ----  | ----  | ----  | ----  | ----  |
| CE | 79.75 ± 0.46 | 90.79 ± 0.04 | 70.13 ± 0.32  | 88.63 ± 0.22 |
| LDAM + DRW | 86.15 ± 0.16 | 91.17 ± 0.10 | 84.48 ± 0.38  | 91.20 ± 0.04 |
| VS | 86.29 ± 0.13 | 91.75 ± 0.09 | 85.04 ± 0.16 | 91.68 ± 0.08 |
| VS + ADRW + TLA (Ours) | **86.42 ± 0.10** | **91.82 ± 0.16** | **85.55 ± 0.22**  | **91.80 ± 0.04** |

<!-- - 200 Epochs w/o SAM and RandAug -->

## Citation

```
@InProceedings{ddc,
    title = {A Unified Generalization Analysis of Re-Weighting and Logit-Adjustment for Imbalanced Learning},
    author = {Zitai Wang and  Qianqian Xu and Zhiyong Yang and Yuan He and Xiaochun Cao and Qingming Huang},
    booktitle = {Annual Conference on Neural Information Processing Systems},
    year = {2023},
}
```
