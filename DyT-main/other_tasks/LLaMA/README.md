# LLaMA with DyT

This guide provides instructions for reproducing the LLaMA results with our proposed DynamicTanh (DyT) modifications, as presented in our paper. We use the [fms-fsdp](https://github.com/foundation-model-stack/fms-fsdp/tree/main) framework to train our LLaMA models. Follow the steps below to reproduce the results.

## 1. Set Up the Python Environment

Follow the original [fms-fsdp](https://github.com/foundation-model-stack/fms-fsdp/tree/main) documentation to set up the required Python environment for the project.

You'll need one additional Python library to save dataloader checkpoints in case you need to resume training without repeating the training data:

```bash
pip install torchdata
```

## 2. Generate Tokenized Data
Since the original fms-fsdp repository does not provide a standard dataset, we used the Pile dataset. To simplify the process, we've included a script that generates tokenized Arrow files from the original dataset.
First, determine the world size you want to use, as this will dictate the total number of Arrow files generated. In general, you should create the same number of Arrow files as the maximum total number of GPUs you plan to use for model training.
After deciding on your world size, run the following command:
```bash
python prepare_data.py \
  --rank $RANK \
  --world_size $WORLD_SIZE \
  --data_path /path/to/data \
  --output_path /path/to/output_dir \
  --max_num_tokens $MAX_NUM_TOKENS \
  --tokenizer $TOKENIZER
```

</details>

**Important**: You need to run this command `$WORLD_SIZE` times, each with a different `$RANK` ranging from `0` to `$WORLD_SIZE - 1`.

For large datasets where you only want to tokenize a subset, set `$MAX_NUM_TOKENS` appropriately. The total tokens in your tokenized data will be `$MAX_NUM_TOKENS × $WORLD_SIZE`. For the tokenizer, we use "meta-llama/Llama-2-7b-chat-hf".

## 3. Run the Experiments

Below are the commands for training various sizes of LLaMA models with DyT.


<details>
<summary>LLaMA2 7B Training Command</summary>

```bash
MODEL_ARGS="\
--model_variant=llama2_7b \
--ckpt_load_path=/checkpoint/path \
--ckpt_save_path=/checkpoint/path \
--data_path=/dataset/path \
--sharding_strategy=hsdp \
--fsdp_activation_checkpointing=False \
--selective_checkpointing=1 \
--mixed_precision=True \
--low_cpu_fsdp=True \
--batch_size=2 \
--learning_rate=0.0003 \
--checkpoint_interval=5000 \
--tracker=wandb \
--tracker_dir=/tracker/path \
--tracker_project_name=tracker_project_name \
--tracker_run_name=llama2_dyt_7b \
--attn_alpha_init_value=0.8 \
--ffn_alpha_init_value=0.2 \
--dec_alpha_init_value=0.2 
"
srun torchrun --nnodes=64 --nproc_per_node=8 main_training_llama.py ${MODEL_ARGS}
```

</details>


<details>
<summary>LLaMA2 13B Training Command</summary>

```bash
MODEL_ARGS="\
--model_variant=llama2_13b \
--ckpt_load_path=/checkpoint/path \
--ckpt_save_path=/checkpoint/path \
--data_path=/dataset/path \
--sharding_strategy=hsdp \
--fsdp_activation_checkpointing=True \
--selective_checkpointing=0.5 \
--mixed_precision=True \
--low_cpu_fsdp=True \
--batch_size=2 \
--learning_rate=0.0003 \
--checkpoint_interval=2000 \
--tracker=wandb \
--tracker_dir=/tracker/path \
--tracker_project_name=tracker_project_name \
--tracker_run_name=llama2_dyt_13b \
--attn_alpha_init_value=0.6 \
--ffn_alpha_init_value=0.15 \
--dec_alpha_init_value=0.15 
"
srun torchrun --nnodes=64 --nproc_per_node=8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:54224 main_training_llama.py ${MODEL_ARGS}
```

</details>



<details>
<summary>LLaMA2 34B Training Command</summary>

```bash
MODEL_ARGS="\
--model_variant=llama2_34b \
--ckpt_load_path=/checkpoint/path \
--ckpt_save_path=/checkpoint/path \
--data_path=/dataset/path \
--sharding_strategy=fsdp \
--fsdp_activation_checkpointing=True \
--selective_checkpointing=0.5 \
--mixed_precision=True \
--low_cpu_fsdp=True \
--batch_size=1 \
--learning_rate=0.00015 \
--checkpoint_interval=2000 \
--tracker=wandb \
--tracker_dir=/tracker/path \
--tracker_project_name=tracker_project_name \
--tracker_run_name=llama2_dyt_34b \
--attn_alpha_init_value=0.2 \
--ffn_alpha_init_value=0.05 \
--dec_alpha_init_value=0.05 
"
srun torchrun --nnodes=128 --nproc_per_node=8 main_training_llama.py ${MODEL_ARGS}
```

</details>



<details>
<summary>LLaMA2 70B Training Command</summary>

```bash
MODEL_ARGS="\
--model_variant=llama2_70b \
--ckpt_load_path=/checkpoint/path \
--ckpt_save_path=/checkpoint/path \
--data_path=/dataset/path \
--sharding_strategy=fsdp \
--fsdp_activation_checkpointing=True \
--selective_checkpointing=1 \
--mixed_precision=True \
--low_cpu_fsdp=True \
--batch_size=1 \
--learning_rate=0.00015 \
--checkpoint_interval=2000 \
--tracker=wandb \
--tracker_dir=/tracker/path \
--tracker_project_name=tracker_project_name \
--tracker_run_name=llama2_dyt_70b \
--attn_alpha_init_value=0.2 \
--ffn_alpha_init_value=0.05 \
--dec_alpha_init_value=0.05 
"
srun torchrun --nnodes=128 --nproc_per_node=8 main_training_llama.py ${MODEL_ARGS}
```

</details>

To reproduce the baseline results, follow the original [fms-fsdp](https://github.com/foundation-model-stack/fms-fsdp) repository using the same command, excluding the last three arguments, which are specific to DyT.


## Acknowledgement
This repository is built using the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack) library and [fms-fsdp](https://github.com/foundation-model-stack/fms-fsdp) repository.
