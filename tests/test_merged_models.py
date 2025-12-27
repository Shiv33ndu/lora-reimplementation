import torch
from lora.lora_linear import LoRALinear



def test_lora_merge_equivalence():
    # ========================== Correctness check code =================================
    # 
    # We check the sanity of merging by comparing the outputs before and after merging 
    # the LoRA weights.
    # 
    # The goal is to ensure that the model behaves identically before and after the merge.
    # 
    # ===================================================================================
    
    torch.manual_seed(0)

    layer = LoRALinear(10, 6, r=4, alpha=16)

    x = torch.randn(3, 10)

    # simulated_training
    layer.A.data.normal_()
    layer.B.data.normal_()

    y_before = layer(x)

    # merge the layer 
    layer.merge()

    y_after = layer(x)

    assert torch.allclose(y_before, y_after, atol=1e-5)