from tests.utils import DummyRoberta
from inject_lora.patch_roberta import inject_lora_into_roberta, freeze_non_lora_params


# freeze_non_lora_params(model)





def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_roberta_lora_injection():
    """
    This method checks the LoRA Layer injection into base models
    by counting total parameters and trainable parameters.

    pass:
        if the count of trainable parameters fall into the range
        0.1% - 0.3% of total parameters, as per LoRA paper.
    """

    # for testing purpose rely on DummyRoberta Model, it approximates same parameter count
    model = DummyRoberta(num_layers=12, hidden_size=768)
    model = inject_lora_into_roberta(model, r=4, alpha=16)
    freeze_non_lora_params(model)

    total_params = count_parameters(model)

    trainable_params = count_trainable_parameters(model)

    trainable_percentage = (trainable_params / total_params) * 100

    print(f"Total Params: {total_params} | Trainable Params : {trainable_params}")
    print(f"Trainable parameters percentage: {trainable_percentage:.2f}%")

    # range verification for trainable params
    assert 0.1 <= trainable_percentage <= 0.3, f"Trainable parameters percentage {trainable_percentage:.2f}% is not within the expected range (0.1% to 0.3%)"

    for layer in model.encoder.layer:
        assert hasattr(layer.attention.self.query, "A")
        assert hasattr(layer.attention.self.query, "B")

        assert hasattr(layer.attention.self.value, "A")
        assert hasattr(layer.attention.self.value, "B")

        # Ensure key is untouched
        assert not hasattr(layer.attention.self.key, "A")
        assert not hasattr(layer.attention.self.key, "B")

    print("LoRA injection test passed ✅")

test_roberta_lora_injection()