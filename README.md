# LoRA (Low Rank Adaptation) Re-implementation

This project re-implements LoRA (Hu et al., 2021) paper from scratch.
I demonstrate that low-rank adaptation achieves comparable performance with 95% fewer parameters to train.



## Repo Directory structure 

```csharp
LoRA-from-scratch/
├── lora/
│   ├── lora_linear.py
│   ├── patch_roberta.py
│   ├── patch_gpt2.py
├── experiments/
│   ├── train_roberta.py
│   ├── train_gpt2.py
├── results/
│   ├── plots.ipynb
├── README.md
```


## Injecting LoRALinear into RoBERTa

As per the paper, the best trade-off to apply LoRA is on the $W_q$ and $W_v$ matrices.

Let's look at the overview of HUggingFace RoBERTa's Anatomy

```yaml
RobertaModel
 └── encoder
     └── layer[i]
         └── attention
             └── self
                 ├── query : nn.Linear
                 ├── key   : nn.Linear
                 ├── value : nn.Linear
```

We only going to touch: 
- `layer.attention.self.query`
- `layer.attention.self.value`

Rest of the matrices update doesn't add up much into performance in downstream task adaptation.