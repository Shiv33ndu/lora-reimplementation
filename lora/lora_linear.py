import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: float, bias: bool = True):
        """
        LoRA-enhanced Linear Layer
        
        Args:
            :param in_features: input dimension
            :type in_features: int
            :param out_features: output dimension
            :type out_features: int
            :param r: rank of LoRA update
            :type r: int
            :param alpha: LoRA scaling factor
            :type alpha: float
            :param bias: Whether to use bias
            :type bias: bool
        """

        super().__init__()

        # ------ Original (frozen) Weight ------
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=False
            )  
        
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features),
                requires_grad=False
            )
        else:
            self.bias = None

        
        # -------- LoRA Parameters --------
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # A : with dim (r x in_features)
        self.A = nn.Parameter(torch.empty(r, in_features))

        # B : with dim (out_features x r)
        self.B = nn.Parameter(torch.zeros(out_features, r))

        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize parameters following the LoRA paper
        
        """

        # standard Linear init for forzen weight
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # A ~ N(0, σ^2)
        nn.init.normal(self.A, mean=0.0, std=0.02)

        # B = 0 (so ΔW = 0 at start)
        nn.init.zeros_(self.B)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:

        y = W0 . x + (B A x) * ( alpha / r) 
        
        Args:
        :param x: Input 
        :type x: torch.Tensor

        Return:
        :return: Returns the computed y
        :rtype: Tensor
        """

        # Base projection
        result = x @ self.weight.T

        # LoRA update 
        if self.r > 0:
            lora_update = (x @ self.A.T) @ self.B.T
            result = result + self.scaling * lora_update

        if self.bias is not None:
            result += self.bias
    
        return result


    def merge(self):
        """
        Merge LoRA weights (BA) into the frozen base weight.
        After merge, the layer behaves as a standard nn.Linear.
        """

        if self.r > 0:
            # computing ΔW = (alpha / r) * B @ A
            delta_w = self.scaling * (self.B @ self.A)

            # merge back to original weight
            self.weight.data += delta_w

            # once the merge is done, we dont need B & A
            self.A.requires_grad = False 
            self.B.requires_grad = False 

            # purge LoRA weights to save memory
            del self.A
            del self.B

            self.r = 0

# layer = LoRALinear(10, 5, r=4, alpha=16)
# x = torch.randn(2,10)

# y1 = layer(x)

# # check only A and B are trainable 
# for name, p in layer.named_parameters():
#     print(name, p.requires_grad)