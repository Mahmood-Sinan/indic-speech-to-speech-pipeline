# Indic Voice Agent

A real-time, spoken-language voice assistant for Indian languages. It listens
on the microphone, detects when you're speaking, transcribes your speech,
sends it to Gemini, and speaks the reply back — all in a continuous loop.

```
mic -> Silero VAD (detect speech) -> IndicConformer (speech-to-text)
    -> Gemini (LLM)
    -> Indic Parler-TTS (text-to-speech) -> speaker
```

## Requirements

- Linux with a working microphone and speakers (ALSA/PulseAudio/PipeWire)
- Python 3.12
- A Gemini API key ([get one here](https://aistudio.google.com/apikey))
- Optional: an NVIDIA GPU with CUDA for faster TTS (the pipeline runs on
  CPU-only machines too, just slower)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Gemini API key (either name works):

```bash
export GOOGLE_API_KEY="your-api-key"
```

Find your microphone's device index:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Open `pipeline/main_optimized.py` and set `DEVICE` (and `CHANNELS`, if your
mic isn't stereo) to match the input device from that list, and set
`LANGUAGE` to the language you'll speak in (`hi`, `ta`, `te`, `kn`, `ml`,
`bn`, ...). These are hardware/preference specific, so they aren't read
from an env var — just edit the constants at the top of the file.

## Running it

```bash
python pipeline/main_optimized.py
```

Speak into the microphone; the agent detects the start/end of your
utterance via VAD, transcribes it, gets a reply from Gemini, and speaks it
back. Press `Ctrl+C` to stop.

## Project layout

```
pipeline/   the assembled end-to-end voice agent
vad/        voice activity detection experiments
stt/        speech-to-text experiments
tts/        text-to-speech experiments
llm/        LLM (Gemini) experiments
```

Each non-`pipeline` folder holds small standalone scripts used to test one
stage of the pipeline in isolation before it was wired into `pipeline/`.
They're useful for debugging a single stage (e.g. "is my mic detecting
speech?") without running the whole agent.

### `pipeline/`

- **`main_optimized.py`** — the current, working entry point. Loads VAD
  (CPU), IndicConformer STT (CPU), Gemini (API), and Indic Parler-TTS (GPU
  if available) once at startup, then runs a loop: read mic chunks -> VAD
  decides when an utterance starts/ends -> transcribe -> ask Gemini ->
  synthesize speech -> play it back.
- **`main.py`** — an earlier snapshot of the same pipeline, kept for
  reference. `main_optimized.py` is the maintained version (fixed GPU OOM
  from IndicConformer defaulting to CUDA, and TTS audio dtype).

### `vad/` — voice activity detection

- **`mic_test.py`** — records 5s from the default input device and prints
  basic signal stats (min/max/RMS). Use this first to confirm your mic works.
- **`vad_test.py`** — records 5s, then runs Silero VAD's
  `get_speech_timestamps` over the whole clip and prints detected speech
  segments.
- **`realtime_vad.py`** — streams the mic continuously and prints
  `SPEECH START`/`SPEECH END` as Silero VAD's per-chunk probability crosses
  a threshold. This is the streaming logic reused in `pipeline/`.
- **`test_vad.py`** — minimal smoke test that just loads the Silero VAD
  model.

### `stt/` — speech-to-text

- **`stt_test.py`** — records 5s from the mic, saves it to `test.wav`, and
  transcribes it with IndicConformer (CTC decoding).
- **`test_stt_model_load.py`** — minimal smoke test that just loads the
  IndicConformer model and prints which device it landed on.

### `tts/` — text-to-speech

- **`tts_test.py`** — synthesizes a fixed Malayalam sentence with Indic
  Parler-TTS, saves it to `tts_output.wav`, and plays it back.

### `llm/`

- **`llm_test.py`** — sends a single fixed prompt to Gemini and prints the
  response, to verify the API key and client are working.

## Notes / gotchas

- **IndicConformer's STT device.** The `ai4bharat/indic-conformer-*` model
  loads itself with custom (`trust_remote_code=True`) code that picks CUDA
  automatically whenever it's available, ignoring `.to("cpu")`. On small
  GPUs this can silently exhaust VRAM. `pipeline/main_optimized.py` works
  around this by temporarily faking CUDA's absence while the STT model
  loads, so it actually stays on CPU when `STT_DEVICE = "cpu"`.
- **VRAM budget.** On a 6GB GPU, running TTS on CUDA alongside the STT/VAD
  models on CPU is what fits. If you have more VRAM, `TTS_DEVICE` (and the
  above STT workaround) can be adjusted to move more of the pipeline to GPU.
- **Mic device index (`DEVICE` in the scripts) is machine-specific** — it
  will not be the same on your machine. Always check with
  `sd.query_devices()` first.
