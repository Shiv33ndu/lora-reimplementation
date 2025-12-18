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





# ========================== Correctness check code =============================
#  Ucomment this to check correctness after LoRA weight mergeing  
# ===============================================================================
# x = torch.randn(2, 768)

# # Before merge
# model_pre = RobertaForSST2(r=4, alpha=16)
# model_pre.load_state_dict(ckpt["lora"], strict=False)
# out_pre = model_pre.roberta.encoder.layer[0].attention.self.query(x)

# # After merge
# model_post = RobertaForSST2(r=4, alpha=16)
# model_post.load_state_dict(ckpt["lora"], strict=False)
# merge_lora_weights(model_post)
# out_post = model_post.roberta.encoder.layer[0].attention.self.query(x)

# # Compare
# assert torch.allclose(out_pre, out_post, atol=1e-5), "LoRA merge correctness check failed"
# print("LoRA merge correctness check passed ✅")
