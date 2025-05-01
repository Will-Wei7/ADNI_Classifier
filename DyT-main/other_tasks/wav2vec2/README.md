# wav2vec 2.0 with DyT

This guide provides instructions for reproducing the wav2vec 2.0 results with our proposed modifications, as presented in our paper. Follow the steps below to set up the environment, apply the patches, and run the experiments.

## 1. Clone the fairseq Repository

Clone the official fairseq repository from GitHub:
```
git clone https://github.com/facebookresearch/fairseq.git
```

## 2. Set Up the Python Environment

Create and activate a Conda environment with the required dependencies:
```
conda create -n w2v python=3.10
conda activate w2v
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install soundfile

cd fairseq
pip install --editable ./
```

*(Fairseq does not provide a config for wav2vec 2.0 Large with LibriSpeech. We created our own by following the instructions from the original paper.)*
Copy the configuration file for wav2vec 2.0 Large with LibriSpeech:
```
cp wav2vec2_large_librispeech.yaml ./fairseq/examples/wav2vec/config/pretraining/
```

## 3. Apply DynamicTanh Patch (Optional)
*(Skip this step if you want to reproduce the baseline results.)* \
To reproduce the results using Dynamic Tanh (DyT), apply the following patch:
```
cp dynamic-tanh.patch fairseq
cd fairseq
git apply dynamic-tanh.patch
```

## 4. Run Experiments

You can reproduce the dynamic-tanh pretraining results using the following command:

### wav2vec 2.0 Base

```
srun torchrun --nnodes=8 --nproc_per_node=8 fairseq-hydra-train \
    task.data=/path/to/manifest \
    --config-dir ./examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech
```

### wav2vec 2.0 Large

```
srun torchrun --nnodes=16 --nproc_per_node=8 fairseq-hydra-train \
    task.data=/path/to/manifest \
    --config-dir ./examples/wav2vec/config/pretraining \
    --config-name wav2vec2_large_librispeech
```

