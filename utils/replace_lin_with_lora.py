import torch
from lora.lora_linear import LoRALinear

def replace_linear_with_lora(
        linear_layer: torch.nn.Linear,
        r: int,
        alpha: int,
) -> LoRALinear:
    """
    Replaces nn.Linear layers of Transformer's attention layer
    and replaces it with LoRALinear, also:
    - copies the pretrained weights
    - freezes them
    - inserts LoRA matrices AB 
    
    Args:
        linear_layer: Linear Layer from base model | type : torch.nn.Linear
        r: LoRA matrix rank | type: int
        aplha: scaling factor for LoRA | type: int 
    """

    lora_layer = LoRALinear(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        r=r,
        alpha=alpha,
        bias=linear_layer.bias is not None
    )

    # copying the pretrained weights 
    lora_layer.weight.data = linear_layer.weight.data.clone() 

    # if bias exist, copy that too
    if linear_layer.bias is not None:
        lora_layer.bias.data = linear_layer.bias.data.clone()

    # return the injected lora_layer 
    return lora_layer 