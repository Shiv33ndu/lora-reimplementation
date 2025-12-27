from transformers import RobertaModel
from utils.replace_lin_with_lora import replace_linear_with_lora


def inject_lora_into_roberta(
        model: RobertaModel,
        r: int = 4,
        alpha: int = 16
):
    """
    Injects LoRA into Wq and Wv of RoBERTa using a helper function
    replace_linear_with_lora
    
    args:
    model: RoBERTa model
    r: LoRA rank | type: int
    alpha: LoRA scaling factor | type: int
    """

    for layer in model.encoder.layer:
        attn = layer.attention.self

        # replace query with LoRA
        attn.query = replace_linear_with_lora(
            attn.query, r=r, alpha=alpha
        )

        # replace value with LoRA
        attn.value = replace_linear_with_lora(
            attn.value, r=r, alpha=alpha
        )
    
    return model


def freeze_non_lora_params(model):
    """
    This freezes out the original weight matrices
    by giving them False flag for gradient update
    """
    for name, param in model.named_parameters():
        if "A" in name or "B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False



