from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
from lora.patch_roberta import inject_lora_into_roberta, freeze_non_lora_params
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # to speedup the Training on Kaggle

dataset = load_dataset("glue", "sst2")

train_ds = dataset["train"]
val_ds = dataset["validation"]

def get_tokenizer():
    return RobertaTokenizer.from_pretrained("roberta-base")

tokenizer = get_tokenizer()

def tokenize_fn(batch):
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=64  # to deal with OOM on kaggle environment
    )

# tokenize the training and validation set
train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

# set the format of test and validation set
train_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

val_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)


# model and LoRA injection
class RobertaForSST2(nn.Module):
    def __init__(self, r=4, alpha=16):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.roberta = inject_lora_into_roberta(self.roberta, r=r, alpha=alpha)
        freeze_non_lora_params(self.roberta)

        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, label=None):
        outputs = self.roberta(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        cls_rep = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_rep)

        loss = None

        if label is not None:
            loss = nn.CrossEntropyLoss()(logits, label)

        return {"loss": loss, "logits": logits}
    

# DataLoader

train_loader = DataLoader(
    train_ds, batch_size=32, shuffle=True
)

val_loader = DataLoader(
    val_ds, batch_size=64
)


# Optimizer (LoRA + classifier only)

def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

model = RobertaForSST2(r=4, alpha=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(
    get_trainable_params(model),
    lr=1e-4,
    weight_decay=0.01
)


# Training loop epoch

def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = {k: v.to(device) for k,v in batch.items()}
        optimizer.zero_grad()

        with autocast():
            out = model(**batch)
            loss = out["loss"]
    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            total_loss += loss.item()
    
    return total_loss / len(loader)


# evaluation for accuracy
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k,v in batch.items()}
            out = model(**batch)
            preds = out["logits"].argmax(dim=-1)

            correct += (preds == batch["label"]).sum().item()
            total += batch["label"].size(0)
    
    return correct / total



# ========================================
#                Training
# ========================================

if __name__ == "__main__":
    EPOCHS = 3

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader)
        val_acc = evaluate(model, val_loader)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
        )