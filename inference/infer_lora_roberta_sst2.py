import torch
from utils.load_model import load_lora_roberta
from experiments.train_lora_roberta_sst2 import get_tokenizer

id2label = {
    0: "Negative",
    1: "Positive"
}

ckpt_path = "checkpoints/lora_roberta_sst2.pt"

model = load_lora_roberta(ckpt_path, "cpu")

tokenizer = get_tokenizer()

inputs = tokenizer(
    "This movie was absolutely trash!",
    return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs)["logits"]
    print(logits.argmax(dim=-1))
    pred_id = logits.argmax(dim=-1).item()
    print(id2label[pred_id])