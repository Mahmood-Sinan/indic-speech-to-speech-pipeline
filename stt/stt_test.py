import sounddevice as sd
import soundfile as sf
import torch
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

SAMPLE_RATE = 16000 # takes these much microphone signals per sec, why 16k ? stt models expects that much.
CHANNELS = 1 # 1 means mono, 2 means stereo. If using stereo we have to avg the signal(signal = (left+right)/2)
DURATION = 5
DEVICE = 4  # This is the audio device the sounddevice should use. 
            # Soundcard uses this interface to write DMIC output. Depends on system.

# Language code
# Examples:
# hi = Hindi
# ta = Tamil
# te = Telugu
# kn = Kannada
# ml = Malayalam
# bn = Bengali
LANGUAGE = "ml"

print("Loading IndicConformer...")

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

print("Model loaded.")
print("Device:", model.d)

# -----------------------------
# Record microphone
# -----------------------------

print(f"\nSpeak in {LANGUAGE} for {DURATION} seconds...")

#sd means sounddevice
audio = sd.rec(
    int(DURATION * SAMPLE_RATE), # total number of samples to record
    samplerate=SAMPLE_RATE, # samples per sec
    channels=CHANNELS,
    dtype="float32", # representation of audio sample in memory
    device=DEVICE, # use this audio device interface to record(means whatever gets written here, capture)
)

sd.wait() # wait for recording to finish

audio = audio[:, 0] # reshaping because the shape of the audio will be [total samples, 1].
                    # Reshape it to [total samples]

# -----------------------------
# Save WAV
# -----------------------------

wav_path = "test.wav"

sf.write(
    wav_path,
    audio,
    SAMPLE_RATE,
)

print(f"Saved recording to {wav_path}")

# -----------------------------
# Convert to tensor
# -----------------------------

wav = torch.from_numpy(audio).unsqueeze(0).to(model.d)
# Model expects a torch tensor rather than np array so we do torch.from_numpy(audio)
# Model expects a 2D array in [1, total samples] dimension. So we do unsqueeze(0) means 
#   add an extra dimension in 0th place
# Model expects audio to be in same memory as where model is running, so we do to(model.d)

print("\nAudio tensor:")
print("Shape:", wav.shape)
print("Device:", wav.device)
print("Duration:", wav.shape[-1] / SAMPLE_RATE, "seconds") # wave.shape[-1] returns total no of
# samples, so wav.shape/sample rate gives no of seconds

# -----------------------------
# Transcribe
# -----------------------------

print("\nTranscribing...")

with torch.inference_mode():
    output = model(
        wav,
        LANGUAGE,
        "ctc", # some kind of encoding
    )

print("\nResult:")
print(output)