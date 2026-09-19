import torch
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

print("Loading model...")

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

print("Model loaded successfully.")
print("CUDA available:", torch.cuda.is_available())
