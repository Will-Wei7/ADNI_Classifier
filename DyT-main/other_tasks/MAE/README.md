# Masked Autoencoders (MAEs) with DyT

This guide provides instructions for reproducing the MAE results with our proposed modifications, as presented in our paper. Follow the steps below to set up the environment, apply the patches, and run the experiments.

## 1. Clone the MAE Repository

Clone the official MAE repository from GitHub:
```
git clone https://github.com/facebookresearch/mae.git
```

## 2. Set Up the Python Environment

The original repository relies on outdated dependencies that may be incompatible with newer GPUs. We have updated the dependencies to ensure compatibility while preserving the integrity of the original implementation.

Set up the Python environment with the following commands:
```
conda create -n MAE python=3.12
conda activate MAE
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install timm==1.0.15 tensorboard
```

## 3. Apply Compatibility Fix

Update the original MAE code for compatibility by applying the provided patch:
```
cp compatibility-fix.patch mae
cd mae
git apply compatibility-fix.patch
```

## 4. Apply DynamicTanh Patch (Optional)
*(Skip this step if you wish to reproduce the baseline results.)* \
To reproduce the results using Dynamic Tanh (DyT), apply the following patches:
```
cp dynamic_tanh.py mae
cp dynamic-tanh.patch mae
cd mae
git apply dynamic-tanh.patch
```

## 4. Run Experiments

After applying the patch, run the MAE pretraining with the following command:
```
torchrun --nnodes=8 --nproc_per_node=8 main_pretrain.py \
    --output_dir /path/to/saving_dir \
    --batch_size 64 \
    --model $MODEL \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path /path/to/imagenet
```
Replace `$MODEL` with one of the following options:
- `mae_vit_base_patch16` - base model.
- `mae_vit_large_patch16` - large model.



## 5. Evaluation

For fine-tuning and evaluation of pretrained models, refer to the original MAE documentation: [FINETUNE](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md).
