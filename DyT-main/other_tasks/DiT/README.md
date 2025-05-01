# Diffusion Transformers (DiTs) with DyT

This guide provides instructions for reproducing the DiT results with our proposed modifications, as presented in our paper. Follow the steps below to set up the environment, apply the patches, and run the experiments.

## 1. Clone the DiT Repository

Clone the official DiT repository from GitHub:
```
git clone https://github.com/facebookresearch/DiT.git
```

## 2. Set Up the Python Environment

Set up the Python environment with the following commands:
```
conda create -n DiT python=3.12
conda activate DiT
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install timm==1.0.15 diffusers==0.32.2 accelerate==1.4.0
```

## 3. Apply Learning Rate Fix

Update the original DiT code to accept learning rate argument by applying the provided patch:
```
cp learning-rate-fix.patch DiT
cd DiT
git apply learning-rate-fix.patch
```

## 4. Apply DynamicTanh Patch (Optional)
*(Skip this step if you wish to reproduce the baseline results.)* \
To reproduce the results using Dynamic Tanh (DyT), apply the following patches:
```
cp dynamic_tanh.py DiT
cp dynamic-tanh.patch DiT
cd DiT
git apply dynamic-tanh.patch
```

## 5. Run Experiments

After applying the patches, run the DiT pretraining with the following command:
```
srun torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --model $MODEL \
    --lr $LEARNING_RATE \
    --data-path /path/to/imagenet/train \
    --results-dir /path/to/saving_dir
```
Replace `$MODEL` with one of the following options: `DiT-B/4`, `DiT-L/4`, or `DiT-XL/2`.
Repace `$LEARNING_RATE` with one of the following options: `1e-4`, `2e-4`, or `4e-4`.


## 6. Evaluation

Follow the [DiT evaluation guide](https://github.com/facebookresearch/DiT) to: 
- Generate samples
- Compute evaluation metrics using the TensorFlow evaluation suite provided in the repository.

