# Efficiency of DyT

This guide provides instructions for reproducing the latency benchmarks reported in Section 6.1 of the original paper.

## 1. Set Up the Python Environment
Set up the Python environment using the following commands:
```
conda create -n DyT python=3.12
conda activate DyT
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install transformers
```

## 2. Benchmark Latency

After setting up the environment, run the benchmark script using the following command:
```
python benchmark.py --layer $LAYER --training
```
Replace `$LAYER` with one of the following options:
- `DyT` - DynamicTanh
- `RMSNorm` - RMSNorm
- `Identity` - Identity

To benchmark latency for the forward pass only, omit the `--training` flag.


## 3. Notes and Limitations

This benchmark is preliminary and does not include any optimization tricks for the forward or backward pass. Therefore, the results should be interpreted as indicative rather than conclusive.

