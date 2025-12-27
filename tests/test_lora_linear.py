from lora.lora_linear import LoRALinear
import torch

def test_lora_zero_init_equivalence():
    """
    This method tests the LoRALinear Layer while keeping the 
    B initialized as matrix of zeros & A as empty matrix/tensor
    and Compares the outputs of nn.Linear and LoRALinear.

    Pass:
        If tests passes the output of lora(x) and base(x) would be
        close enough (within tolerance of 1e-6)  
    """

    torch.manual_seed(0)

    in_feature, out_feature = 8, 4
    x = torch.randn(2, in_feature)

    base = torch.nn.Linear(in_feature, out_feature, bias=False)
    lora = LoRALinear(in_feature, out_feature, r=4, alpha=16, bias=False)

    # copy base weights to perform forward with same weights
    lora.weight.data = base.weight.data.clone()

    y_base = base(x)
    y_lora = lora(x)

    assert torch.allclose(y_base, y_lora, atol=1e-6)


def test_lora_trainable_params_only():
    """
    This method checks the very promise of LoRA paper
    that the trainable params are just A and B LoRA matrices

    We check this using the attribute requires_grad
    """

    layer = LoRALinear(10, 5, r=2, alpha=8)

    trainable = {
        name for name, p in layer.named_parameters() if p.requires_grad
    }

    assert trainable == {"A", "B"}
