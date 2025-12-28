"""
This util is used to create a Dummy RoBERTa Model architecture for testing purpose 
"""

import torch.nn as nn

# # Dummy self-attention layer
# class DummySelfAttention(nn.Module):
#     def __init__(self, hidden_size=768):
#         super().__init__()
#         self.query = nn.Linear(hidden_size, hidden_size)
#         self.key = nn.Linear(hidden_size, hidden_size)
#         self.value = nn.Linear(hidden_size, hidden_size)

# # Dummy feed-forward network (FFN)
# class DummyFFN(nn.Module):
#     def __init__(self, hidden_size=768, intermediate_size=3072):
#         super().__init__()
#         self.intermediate = nn.Linear(hidden_size, intermediate_size)
#         self.output = nn.Linear(intermediate_size, hidden_size)

# # Dummy transformer layer
# class DummyTransformerLayer(nn.Module):
#     def __init__(self, hidden_size=768, intermediate_size=3072):
#         super().__init__()
#         self.attention = nn.Module()
#         self.attention.self = DummySelfAttention(hidden_size)
#         self.ffn = DummyFFN(hidden_size, intermediate_size)

# # Dummy encoder stack
# class DummyEncoder(nn.Module):
#     def __init__(self, num_layers=12, hidden_size=768, intermediate_size=3072):
#         super().__init__()
#         self.layer = nn.ModuleList(
#             [DummyTransformerLayer(hidden_size, intermediate_size) for _ in range(num_layers)]
#         )

# # Dummy RoBERTa Model
# class DummyRoberta(nn.Module):
#     def __init__(self, num_layers=12, hidden_size=768, intermediate_size=3072):
#         super().__init__()
#         self.encoder = DummyEncoder(num_layers, hidden_size, intermediate_size)

import torch.nn as nn

# Dummy self-attention layer
class DummySelfAttention(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

# Dummy feed-forward network (FFN)
class DummyFFN(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)

# Dummy transformer layer
class DummyTransformerLayer(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = DummySelfAttention(hidden_size)
        self.ffn = DummyFFN(hidden_size, intermediate_size)

# Dummy encoder stack
class DummyEncoder(nn.Module):
    def __init__(self, num_layers=12, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.layer = nn.ModuleList(
            [DummyTransformerLayer(hidden_size, intermediate_size) for _ in range(num_layers)]
        )

# Dummy RoBERTa Model
class DummyRoberta(nn.Module):
    def __init__(self, num_layers=12, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.encoder = DummyEncoder(num_layers, hidden_size, intermediate_size)


# test_version of fn inject_lora_into_roberta
from utils.replace_lin_with_lora import replace_linear_with_lora


def inject_lora_into_attention_stack(
    model: DummyRoberta,
    r: int,
    alpha: int
):
    """
    Core LoRA injection logic.
    Works on any model that exposes:
    encoder.layer[i].attention.self.{query,value}
    """
    for layer in model.encoder.layer:
        attn = layer.attention.self

        attn.query = replace_linear_with_lora(
            attn.query, r=r, alpha=alpha
        )

        attn.value = replace_linear_with_lora(
            attn.value, r=r, alpha=alpha
        )
    return model

# test version of the fn freeze_non_lora_params 
def freeze_non_lora_params_test(model: DummyRoberta):
    """
    This freezes out the original weight matrices
    by giving them False flag for gradient update
    """
    if model is None:
        raise ValueError("Model is None. Please ensure the model is correctly instantiated.")
    
    for name, param in model.named_parameters():
        if "A" in name or "B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False