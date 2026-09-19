import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5

print("Default input device:")
print(sd.query_devices(kind="input"))

print("\nRecording...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32",
)

sd.wait()

audio = audio[:, 0]

print("\nRecording finished.")
print("Samples:", len(audio))
print("Min:", audio.min())
print("Max:", audio.max())
print("Peak:", np.max(np.abs(audio)))
print("RMS:", np.sqrt(np.mean(audio ** 2)))