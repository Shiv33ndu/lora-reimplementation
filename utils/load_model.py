import torch
from experiments.train_lora_roberta_sst2 import RobertaForSST2

def load_lora_roberta(ckpt_path, device="cpu"):
    """
    Loads Base model and adds Lora weights onto it for reference
    on downstream task

    args:
        ckpt_path: checkpoint file | type :  pytorch serialized .pt
        device: cpu/cuda | type: string

    return:
        model: base model with LoRA weights added to it
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    model = RobertaForSST2(
        r=ckpt["config"]["r"],
        alpha=ckpt["config"]["alpha"]
    )
    model.load_state_dict(ckpt["lora"], strict=False)
    model.eval()

    return model