import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')
import transformers
from dynamic_tanh import convert_rms_to_dyt, convert_rms_to_identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the latency of a LLaMA-2 7B.")
    parser.add_argument("--layer", default="DyT", help="The layer to benchmark.")
    parser.add_argument("--training", action="store_true", help="Whether to benchmark training.")
    args = parser.parse_args()

    assert args.layer.lower() in ["dyt", "identity", "rmsnorm"]
    
    model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    if args.layer.lower() == "dyt":
        model = convert_rms_to_dyt(model)
    elif args.layer.lower() == "identity":
        model = convert_rms_to_identity(model)
    elif args.layer.lower() == "rmsnorm":
        pass
    else:
        raise ValueError("Invalid layer. Must be dyt, identity, or rmsnorm.")
    print(model)
    
    model.to(device=0, dtype=torch.bfloat16)

    samples = []
    for _ in range(200):
        samples.append(torch.randint(0, 32000, (1, 4096), dtype=torch.long, device=0))
    
    torch.cuda.synchronize()
    if args.training:
        for sample in samples[:100]:
            out = model(sample)
            loss = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), sample.view(-1))
            loss.backward()
    else:
        for sample in samples[:100]:
            with torch.no_grad():
                out = model(sample)
                loss = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), sample.view(-1))
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    time_1 = time.time()
    if args.training:
        for sample in samples[100:]:
            out = model(sample)
            loss = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), sample.view(-1))
            loss.backward()
    else:
        for sample in samples[100:]:
            with torch.no_grad():
                out = model(sample)
                loss = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), sample.view(-1))
    torch.cuda.synchronize()
    time_2 = time.time()

    print(f"{args.layer}, {'inference' if not args.training else 'training'} Time: {time_2 - time_1:.2f} seconds")


