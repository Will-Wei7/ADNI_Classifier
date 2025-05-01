import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x)


def convert_rms_to_dyt(module):
    module_output = module
    if isinstance(module, LlamaRMSNorm):
        module_output = DynamicTanh(normalized_shape=module.weight.shape[0])
    for name, child in module.named_children():
        module_output.add_module(name, convert_rms_to_dyt(child))
    del module
    return module_output


def convert_rms_to_identity(module):
    module_output = module
    if isinstance(module, LlamaRMSNorm):
        module_output = nn.Identity()
    for name, child in module.named_children():
        module_output.add_module(name, convert_rms_to_identity(child))
    del module
    return module_output
