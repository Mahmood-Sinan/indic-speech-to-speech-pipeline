import sounddevice as sd
import torch
from silero_vad import load_silero_vad

# -----------------------------
# Configuration
# -----------------------------
SAMPLE_RATE = 16000
CHANNELS = 1

# 512 samples at 16 kHz = 32 ms
CHUNK_SIZE = 512 # the amount of samples processed in one go.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# -----------------------------
# Load Silero VAD
# -----------------------------
model = load_silero_vad() # load VAD silero model
model = model.to(device) # move model to selected device(either gpu or cpu)

print("Silero VAD loaded.")
print("Listening... Press Ctrl+C to stop.\n")

# -----------------------------
# State
# -----------------------------
is_speaking = False


# -----------------------------
# Audio callback, callback functions are called whenever a specified event happens, here it is
# whenever a new chunk of audio arrives.
# -----------------------------
def audio_callback(indata, frames, time, status):
    global is_speaking

    if status:
        print("Audio status:", status)

    # (512, 1) -> (512,)
    audio = indata[:, 0]
    # indata means input data. Here we are converting mono audio input which has 
    # dimension [samples x 1] into [samples] dimension

    # NumPy -> PyTorch
    audio_tensor = torch.from_numpy(audio).to(device)
    # Model expects a torch tensor rather than np array so we do torch.from_numpy(audio)
    # Model expects audio to be in same memory as where model is running, so we do to(model.d)

    # Get speech probability
    speech_probability = model(audio_tensor, SAMPLE_RATE).item()
    # Model calculates the probability of having speech in the audio

    # Simple threshold
    speech_detected = speech_probability > 0.5

    # Detect transitions
    if speech_detected and not is_speaking:
        is_speaking = True
        print("SPEECH START")

    elif not speech_detected and is_speaking:
        is_speaking = False
        print("SPEECH END")


# -----------------------------
# Start microphone stream
# -----------------------------
try:
    # Tells sounddevice to open the microphone and continuously provide audio to audio_callback()
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
    ):
        while True:
            sd.sleep(100) # puts the current python thread into sleep, not sd(the name is confusing)

except KeyboardInterrupt:
    print("\nStopped.")
