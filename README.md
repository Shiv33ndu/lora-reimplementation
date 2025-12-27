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

---

## LoRA on RoBERTa-base (SSt-2) Training

- Total Params : 124793088
- Trainable Params : 147456 (0.12%)
- Ranks (r) : 4
- Alpha (α) : 16
- Sequence Length: 64
- AMP enabled


| Epoch | Train Loss | Val Accuracy |
|------|-----------|--------------|
| 1 | 0.2850 | 93.12% |
| 2 | 0.2162 | 93.00% |
| 3 | 0.1985 | 92.66% |

LoRA weights (A, B) and classifier head are saved and reusable.


---


## Rank Ablation Study (LoRA on SST-2)

To analyze the effect of LoRA rank \( r \) on downstream performance, I conduct a rank
ablation study following the methodology discussed in Section 7.2 of the LoRA paper.

The goal is to evaluate whether task-specific adaptation saturates at low rank, which would
support the **low intrinsic dimension** hypothesis.

<p align="center">
  <img src="results/roberta_sst2_rank_ablation/rank_ablation.png" width="500"/>
</p>

### Experimental Setup
- Base model: RoBERTa-base
- Task: SST-2 (GLUE)
- LoRA applied to: $W_q$  and $W_v$
- Ranks evaluated: $r$ $\in$ ${1, 2, 4, 8\}$
- α = 16, epochs = 3
- Sequence length = 64, AMP enabled
- Only LoRA parameters (A, B) and the classifier head are trainable

### Results

| Rank (r) | Validation Accuracy |
|---------|---------------------|
| 1 | 93.35% |
| 2 | **93.69%** |
| 4 | 92.89% |
| 8 | 93.46% |

### Observation

- Even very small ranks (r = 1, 2) achieve strong performance.
- Increasing rank beyond r ≈ 2–4 does not yield consistent improvements.
- Performance remains within a narrow band (~0.9% absolute) across all ranks.

The slight non-monotonic behavior is expected on small datasets such as SST-2 and can be
attributed to training variance and noise rather than insufficient model capacity.

Overall, these results indicate that effective task adaptation can be achieved using
**low-rank updates**, consistent with the LoRA paper’s claim that most task-relevant updates
lie in a low-dimensional subspace.

---

## Parameter Efficiency vs Rank

One of the key advantages of LoRA is that the number of trainable parameters
scales **linearly** with the chosen rank ($ r $).

For RoBERTa-base (hidden size = 768, 12 layers) with LoRA applied to \( W_q \) and \( W_v \),
the number of trainable parameters is:

$$
\text{Trainable Params} = 48 \times d \times r = 36{,}864 \times r
$$

The LoRA-specific parameter count scales linearly with rank as:
Trainable LoRA Params ≈ 36,864 × r

An additional ~1.5k parameters come from the task-specific classification head.

### Trainable Parameters by Rank (LoRA + Classifier Head)

| Rank (r) | Trainable Parameters | % of Total Params |
|---------|----------------------|------------------|
| 1 | 38,402 | 0.031% |
| 2 | 75,266 | 0.060% |
| 4 | 148,994 | 0.119% |
| 8 | 296,450 | 0.237% |

*Note: In above Table Trainable parameters include both LoRA adaptation matrices (A, B) and the task-specific classification head.*

Despite increasing the rank by **8×**, the fraction of trainable parameters
remains well below **0.3%** of the full model size.

Combined with the rank ablation results, this highlights that strong downstream
performance can be achieved with **extremely small parameter updates**, reinforcing
the low intrinsic dimension hypothesis behind LoRA.

---

## LoRA Weight Merging for Inference

After training, the low-rank LoRA updates can be merged into the frozen base
weights to eliminate any inference-time overhead.

For a LoRA-adapted linear layer:

W_eff = W₀ + (α / r) · B · A

This merge is performed once, after which the model behaves as a standard
Transformer without additional parameters or computation.

All LoRA weights are merged prior to inference, resulting in:
- zero added latency
- no extra memory usage
- identical outputs compared to the unmerged LoRA model


## Testing

Unit tests focus on LoRA-specific invariants:
- zero-init equivalence
- merge correctness
- parameter freezing
- correct injection points

To keep CI fast and deterministic, tests use lightweight dummy transformer
modules instead of downloading pretrained models.


## 🧪 How to run tests

```python
pip install pytest
pytest -s tests/
``` 

