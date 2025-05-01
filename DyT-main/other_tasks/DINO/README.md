# DINO with DyT

This guide provides instructions for reproducing the DINO results with our proposed modifications, as presented in our paper. Follow the steps below to set up the environment, apply the patches, and run the experiments.

## 1. Clone the DINO Repository

Clone the official DINO repository from GitHub:
```
git clone https://github.com/facebookresearch/dino.git
```

## 2. Set Up the Python Environment

Create and activate a Conda environment with the required dependencies:
```
conda create -n DINO python=3.12
conda activate DINO
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

## 3. Apply DynamicTanh Patch (Optional)
*(Skip this step if you want to reproduce the baseline results.)* \
To reproduce the results using Dynamic Tanh (DyT), apply the following patches:
```
cp dynamic_tanh.py dino
cp dynamic-tanh.patch dino
cd dino
git apply dynamic-tanh.patch
```

## 3. Run Experiments

You can reproduce the DINO pretraining results using the following command:

### ViT-B with Patch Size 16

This configuration follows the arguments from the original DINO documentation at [ViT-B-16](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/args.txt).
```
torchrun --nnodes=4 --nproc_per_node=8 main_dino.py \
    --arch vit_base \
    --patch_size 16 \
    --out_dim 65536 \
    --norm_last_layer true \
    --warmup_teacher_temp 0.04 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 50 \
    --use_fp16 false \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --clip_grad 0.3 \
    --batch_size_per_gpu 32 \
    --epochs 400 \
    --freeze_last_layer 3 \
    --lr 0.00075 \
    --warmup_epochs 10 \
    --min_lr 2e-06 \
    --global_crops_scale 0.25 1.0 \
    --local_crops_scale 0.05 0.25 \
    --local_crops_number 10 \
    --seed 0 \
    --num_workers 10 \
    --optimizer adamw \
    --momentum_teacher 0.996 \
    --use_bn_in_head false \
    --drop_path_rate 0.1 \
    --data_path /path/to/imagenet/train \
    --output_dir /path/to/saving_dir
```

### ViT-B with Patch Size 8

This configuration follows the arguments from the original DINO documentation at [ViT-B-8](https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/args.txt).
```
torchrun --nnodes=22 --nproc_per_node=8 main_dino.py \
    --arch vit_base \
    --patch_size 8 \
    --out_dim 65536 \
    --norm_last_layer true \
    --warmup_teacher_temp 0.03 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 50 \
    --use_fp16 false \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --clip_grad 3.0 \
    --batch_size_per_gpu 6 \
    --epochs 300 \
    --freeze_last_layer 3 \
    --lr 0.0005 \
    --warmup_epochs 10 \
    --min_lr 2e-06 \
    --global_crops_scale 0.25 1.0 \
    --local_crops_scale 0.05 0.25 \
    --local_crops_number 10 \
    --seed 0 \
    --num_workers 10 \
    --optimizer adamw \
    --momentum_teacher 0.996 \
    --use_bn_in_head false \
    --drop_path_rate 0.1 \
    --data_path /path/to/imagenet/train \
    --output_dir /path/to/saving_dir
```


## 5. Evaluation
*(Since DINO does not provide fine-tuning code, we use the MAE code for fine-tuning.)* \
For fine-tuning and evaluation of pretrained models, refer to the original MAE documentation: [FINETUNE](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md).
