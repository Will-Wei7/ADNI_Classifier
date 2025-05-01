# DNA Sequence Modeling with DyT

This guide provides instructions for reproducing the DNA sequence modeling results with our proposed DynamicTanh (DyT) modifications, as presented in our paper.

## 1. Clone the Caduceus Repository

Clone the official Caduceus repository from GitHub:

```bash
git clone https://github.com/kuleshov-group/caduceus.git
```

## 2. Set Up the Python Environment and Datasets

Follow the instructions in the original [Caduceus README](https://github.com/kuleshov-group/caduceus/blob/main/README.md) to:
- Set up the Python environment with required dependencies
- Download and prepare the necessary datasets for DNA sequence modeling

## 3. Apply DynamicTanh (DyT) Patch

*(Skip this step if you want to reproduce the baseline results without DyT modifications.)*

To reproduce the results using Dynamic Tanh (DyT), apply the following patches:

```bash
cp dynamic_tanh.py caduceus/
cp dynamic-tanh.patch caduceus/
cd caduceus
git apply dynamic-tanh.patch
```

## 4. Run Experiments

You can reproduce our DNA Sequence Modeling results using the provided SLURM scripts. You may need to edit these scripts to adapt them to your computing environment.

### Caduceus Model Training

```bash
cd slurm_scripts
sbatch run_pretrain_caduceus.sh
```

### HyenaDNA Model Training

```bash
cd slurm_scripts
sbatch run_pretrain_hyena.sh
```

### Model Evaluation

```bash
cd slurm_scripts
bash wrapper_run_genomics.sh
```

