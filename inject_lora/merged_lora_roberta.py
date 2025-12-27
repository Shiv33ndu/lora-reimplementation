import torch
from experiments.train_lora_roberta_sst2 import RobertaForSST2
from utils.merge_lora import merge_lora_weights

# # loading trained LoRA Model
model = RobertaForSST2(r=4, alpha=16)
ckpt = torch.load("checkpoints/lora_roberta_sst2.pt", map_location="cpu")
model.load_state_dict(ckpt["lora"], strict=False)

# merge LoRA weights
merge_lora_weights(model)

# swtich to evaluation mode
model.eval()

# saving the new fine-tuned model, with merged BA LoRA weights
torch.save(
    model.state_dict(),
    "merged_models/roberta_sst2_merged.pt"
)

print("Model Saved!!")






