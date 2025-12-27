"""
This util is used to create a Dummy RoBERTa Model architecture for testing purpose 
"""

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