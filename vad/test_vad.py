import torch
from silero_vad import load_silero_vad, get_speech_timestamps

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model = load_silero_vad()
model = model.to(device)

print("Silero VAD loaded successfully!")
