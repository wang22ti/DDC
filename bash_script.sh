# CIFAR-100 LT (Imb ratio = 100)
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.2 0.2  --wd 0.0001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.5 0.5  --wd 0.0001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type exp --imb_factor 0.01 --loss_type VS --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0.05 --tau 1.0 --rho 0.3 0.3  --wd 0.0002 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type exp --imb_factor 0.01 --loss_type VS --train_rule ADRW-T --gpu 0 --seed 0 --tro 0.25 --gamma 0 --tau 0.75 --rho 0.2 0.4  --wd 0.0005 --epochs 400 --randaug 1 --resume 'log/cifar100_resnet32_VS_None_exp_0.01_0.0002_1/seed_0_sam_[0.2, 0.2]_args_(0.0, 0.05, 1.0)/320_ckpt.pth.tar'

# CIFAR-100 LT (Imb ratio = 10)
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.1 0.1  --wd 0.0001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type exp --imb_factor 0.1 --loss_type LDAM --train_rule DRW --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.3 0.3  --wd 0.00005 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type exp --imb_factor 0.1 --loss_type VS --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0.05 --tau 0.5 --rho 0.1 0.1  --wd 0.0001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type exp --imb_factor 0.1 --loss_type VS --train_rule ADRW-T --gpu 0 --seed 0 --tro 0.1 --gamma 0 --tau 0.75 --rho 0.1 0.5  --wd 0.0001 --epochs 400 --randaug 1 --resume 'log/cifar100_resnet32_VS_None_exp_0.1_1/seed_0_wd_0.000100_sam_[0.1, 0.1]_args_(0.0, 0.05, 0.5)/320_ckpt.pth.tar'

# CIFAR-100 LT (Imb ratio = 100)
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type step --imb_factor 0.01 --loss_type CE --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.3 0.3  --wd 0.00001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type step --imb_factor 0.01 --loss_type LDAM --train_rule DRW --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.3 0.3  --wd 0.00005 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type step --imb_factor 0.01 --loss_type VS --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 1.0 --rho 0.2 0.2  --wd 0.0001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type step --imb_factor 0.01 --loss_type VS --train_rule ADRW-T --gpu 0 --seed 0 --tro 0.1 --gamma 0 --tau 1.0 --rho 0.2 0.2  --wd 0.0004 --epochs 400 --randaug 1 --resume 'log/cifar100_resnet32_VS_None_step_0.01_1/seed_0_wd_0.000100_sam_[0.2, 0.2]_args_(0.0, 0.0, 1.0)/320_ckpt.pth.tar'

# CIFAR-100 LT (Imb ratio = 10)
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type step --imb_factor 0.1 --loss_type CE --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.1 0.1  --wd 0.00001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type step --imb_factor 0.1 --loss_type LDAM --train_rule DRW --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.4 0.4  --wd 0.00005 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type step --imb_factor 0.1 --loss_type VS --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 1.0 --rho 0.1 0.1  --wd 0.0001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar100 --imb_type step --imb_factor 0.1 --loss_type VS --train_rule ADRW-T --gpu 0 --seed 0 --tro 0.05 --gamma 0 --tau 1.0 --rho 0.1 0.3  --wd 0.0003 --epochs 400 --randaug 1 --resume 'log/cifar100_resnet32_VS_None_step_0.1_1/seed_0_wd_0.000100_sam_[0.1, 0.1]_args_(0.0, 0.0, 1.0)/320_ckpt.pth.tar'

# CIFAR-10 LT (Imb ratio = 100)
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.1 0.1  --wd 0.0001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.4 0.4  --wd 0.00005 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type exp --imb_factor 0.01 --loss_type VS --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0.05 --tau 1.0 --rho 0.1 0.1  --wd 0.0002 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type exp --imb_factor 0.01 --loss_type VS --train_rule ADRW-T --gpu 0 --seed 0 --tro 0.15 --gamma 0 --tau 1.0 --rho 0.1 0.2  --wd 0.0002 --epochs 400 --randaug 1 --resume 'log/cifar10_resnet32_VS_None_exp_0.01_1/seed_0_wd_0.000200_sam_[0.1, 0.1]_args_(0.0, 0.05, 1.0)/320_ckpt.pth.tar'

# CIFAR-10 LT (Imb ratio = 10)
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.1 0.1  --wd 0.0001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type exp --imb_factor 0.1 --loss_type LDAM --train_rule DRW --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.4 0.4  --wd 0.00001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type exp --imb_factor 0.1 --loss_type VS --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 1.0 --rho 0.1 0.1  --wd 0.0001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type exp --imb_factor 0.1 --loss_type VS --train_rule ADRW-T --gpu 0 --seed 0 --tro 0.1 --gamma 0.1 --tau 1.0 --rho 0.1 0.1  --wd 0.0001 --epochs 400 --randaug 1 --resume 'log/cifar10_resnet32_VS_None_exp_0.1_1/seed_0_wd_0.000100_sam_[0.1, 0.1]_args_(0.0, 0.0, 1.0)/320_ckpt.pth.tar'

# CIFAR-10 LT (Imb ratio = 100)
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type step --imb_factor 0.01 --loss_type CE --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.1 0.1  --wd 0.00005 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type step --imb_factor 0.01 --loss_type LDAM --train_rule DRW --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.2 0.2  --wd 0.00005 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type step --imb_factor 0.01 --loss_type VS --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 1.25 --rho 0.1 0.1  --wd 0.0001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type step --imb_factor 0.01 --loss_type VS --train_rule ADRW-T --gpu 0 --seed 0 --tro 0.1 --gamma 0 --tau 1.0 --rho 0.1 0.2  --wd 0.00005 --epochs 400 --randaug 1 --resume 'log/cifar10_resnet32_VS_None_step_0.01_1/seed_0_wd_0.000100_sam_[0.1, 0.1]_args_(0.0, 0.0, 1.25)/320_ckpt.pth.tar'

# CIFAR-10 LT (Imb ratio = 10)
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type step --imb_factor 0.1 --loss_type CE --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.1 0.1  --wd 0.00005 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type step --imb_factor 0.1 --loss_type LDAM --train_rule DRW --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 0 --rho 0.2 0.2  --wd 0.00005 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type step --imb_factor 0.1 --loss_type VS --train_rule None --gpu 0 --seed 0 --tro 0 --gamma 0 --tau 1.0 --rho 0.1 0.1  --wd 0.0001 --epochs 400 --randaug 1
python cifar_train_sam.py -a resnet32 --dataset cifar10 --imb_type step --imb_factor 0.1 --loss_type VS --train_rule ADRW-T --gpu 0 --seed 0 --tro 0.1 --gamma 0 --tau 1.0 --rho 0.1 0.1  --wd 0.0001 --epochs 400 --randaug 1 --resume 'log/cifar10_resnet32_VS_None_step_0.1_1/seed_0_wd_0.000100_sam_[0.1, 0.1]_args_(0.0, 0.0, 1.0)/320_ckpt.pth.tar'



