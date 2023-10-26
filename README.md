# A Unified Generalization Analysis of Re-Weighting and Logit-Adjustment for Imbalanced Learning

This is a Pytorch implementation of our paper: [A Unified Generalization Analysis of Re-Weighting and Logit-Adjustment for Imbalanced Learning](https://arxiv.org/abs/2310.04752). 

> **Abstract:** Real-world datasets are typically imbalanced in the sense that only a few classes have numerous samples, while many classes are associated with only a few samples. As a result, a naïve ERM learning process will be biased towards the majority classes, making it difficult to generalize to the minority classes. To address this issue, one simple but effective approach is to modify the loss function to emphasize the learning on minority classes, such as re-weighting the losses or adjusting the logits via class-dependent terms. However, existing generalization analysis of such losses is still coarse-grained and fragmented, failing to explain some empirical results. To bridge this gap, we propose a novel technique named data-dependent contraction to capture how these modified losses handle different classes. On top of this technique, a fine-grained generalization bound is established for imbalanced learning, which helps reveal the mystery of re-weighting and logit-adjustment in a unified manner. Furthermore, a principled learning algorithm is developed based on the theoretical insights. Finally, the empirical results on benchmark datasets not only validate the theoretical results but also demonstrate the effectiveness of the proposed method.

Our codes are based on the repositories [Escaping Saddle Points for Effective Generalization on Class-Imbalanced Data](https://github.com/val-iisc/saddle-longtail).

## Dependencies
Please refer to the `requirements.yml` in the root folder.

## Results


## Citation

```
@InProceedings{ddc,
    title = {A Unified Generalization Analysis of Re-Weighting and Logit-Adjustment for Imbalanced Learning},
    author = {Zitai Wang and  Qianqian Xu and Zhiyong Yang and Yuan He and Xiaochun Cao and Qingming Huang},
    booktitle = {Annual Conference on Neural Information Processing Systems},
    year = {2023},
}
```