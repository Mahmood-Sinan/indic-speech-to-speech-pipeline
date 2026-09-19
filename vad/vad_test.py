import sounddevice as sd
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5

# -----------------------------
# Load VAD
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

model = load_silero_vad()
model = model.to(device)

print("Silero VAD loaded.")

# -----------------------------
# Record microphone
# -----------------------------
print("\nSpeak normally for 5 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32",
)

sd.wait()

# Convert (samples, 1) -> (samples,)
audio = audio[:, 0]

print("Recording finished.")

# -----------------------------
# Run VAD
# -----------------------------
audio_tensor = torch.from_numpy(audio).to(device)

speech_timestamps = get_speech_timestamps(
    audio_tensor,
    model,
    sampling_rate=SAMPLE_RATE,
)

# -----------------------------
# Display results
# -----------------------------
print("\nSpeech segments:")

if not speech_timestamps:
    print("No speech detected.")
else:
    for segment in speech_timestamps:
        start = segment["start"] / SAMPLE_RATE
        end = segment["end"] / SAMPLE_RATE

        print(f"  {start:.2f}s -> {end:.2f}s")
