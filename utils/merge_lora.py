def merge_lora_weights(model):
    """
    Recursively merge all LoRALinear layers in the model
    """

    for module in model.modules():
        if hasattr(module, "merge"):
            module.merge()