# Indic Voice Agent

A real-time, spoken-language voice assistant for Indian languages. It listens
on the microphone, detects when you're speaking, transcribes your speech,
sends it to Gemini, and speaks the reply back — all in a continuous loop.

```
mic -> Silero VAD (detect speech) -> IndicConformer (speech-to-text)
    -> local NMT (translate to English)
    -> Gemini (LLM)
    -> local NMT (translate back to your language)
    -> Indic Parler-TTS (text-to-speech) -> speaker
```

The LLM always works in English. A local NMT stage converts text in both
directions around it, so you can speak in your own language and hear the
reply in the same language. It's AI4Bharat's IndicTrans2, ported from
Suno Sutra's `bhashini_models` NMT server as int8-quantized CTranslate2
checkpoints (the same models Bhashini/Suno Sutra run on-device) — see
[`nmt/`](#nmt--local-text-to-text-translation) below. There are no
translation credentials and no network calls at inference time.

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

No credentials are needed for translation, and there's no HuggingFace
download either — the two ~214MB IndicTrans2 checkpoints (int8
CTranslate2 format) are local files under `nmt/checkpoints/`, which are
**not committed to git**. Fetch them from the `suno-sutra-sw` clone (this
requires [git-lfs](https://git-lfs.com/) and pulls ~1.6GB, since the
model files sit inside `bhashini_models.zip` alongside ASR/TTS assets we
don't need):

```bash
cd suno-sutra-sw
git lfs install --local   # one-time, local to this clone
git lfs pull --include="rootfs/roles/indic/files/bhashini_models.zip"
cd ..

mkdir -p nmt/checkpoints
cd /tmp && unzip -o "$OLDPWD/suno-sutra-sw/rootfs/roles/indic/files/bhashini_models.zip" \
  "bhashini_models/nmt/checkpoints/indic-en/ct2_int8_model/*" \
  "bhashini_models/nmt/checkpoints/en-indic/ct2_int8_model/*"
mv bhashini_models/nmt/checkpoints/indic-en "$OLDPWD/nmt/checkpoints/"
mv bhashini_models/nmt/checkpoints/en-indic "$OLDPWD/nmt/checkpoints/"
cd "$OLDPWD"
```

You can point at different checkpoint directories, or run NMT on the GPU
instead of CPU:

```bash
export NMT_INDIC_EN_CHECKPOINT="/path/to/indic-en/ct2_int8_model"
export NMT_EN_INDIC_CHECKPOINT="/path/to/en-indic/ct2_int8_model"
# optional: run NMT on the GPU instead of CPU
# export NMT_DEVICE="cuda"
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
utterance via VAD, transcribes it, translates it to English for Gemini,
translates the reply back, and speaks it. Press `Ctrl+C` to stop.

## Project layout

```
pipeline/   the assembled end-to-end voice agent
vad/        voice activity detection experiments
stt/        speech-to-text experiments
nmt/        local text-to-text translation (Bhashini/IndicTrans2) client + test
tts/        text-to-speech experiments
llm/        LLM (Gemini) experiments
```

Each non-`pipeline` folder holds small standalone scripts used to test one
stage of the pipeline in isolation before it was wired into `pipeline/`.
They're useful for debugging a single stage (e.g. "is my mic detecting
speech?") without running the whole agent.

### `pipeline/`

- **`main_optimized.py`** — the current, working entry point. Loads VAD
  (CPU), IndicConformer STT (CPU), the local NMT translator (CPU), Gemini
  (API), and Indic Parler-TTS (GPU if available) once at startup, then
  runs a loop: read mic chunks -> VAD decides when an utterance
  starts/ends -> transcribe -> translate to English -> ask Gemini ->
  translate back to your language -> synthesize speech -> play it back.
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

### `nmt/` — local text-to-text translation

- **`nmt_client.py`** — the `IndicTransTranslator` class wrapping two
  local CTranslate2 models (one per direction, `*->eng_Latn` and
  `eng_Latn->*`). Runs on CPU by default. Handles the pipeline's short
  language codes (`hi`, `ta`, ...) -> IndicTrans2's script-qualified
  codes (`hin_Deva`, `tam_Taml`, ...) and delegates sentence splitting,
  tokenisation and translation to `bhashini_engine/`. Failures raise
  `NMTError`, which the pipeline catches to keep the turn running with
  untranslated text.
- **`bhashini_engine/`** — AI4Bharat's IndicTrans2 CTranslate2 inference
  engine, vendored from Suno Sutra's `bhashini_models` NMT server (see
  the module docstring in `bhashini_engine/engine.py` for exact
  provenance and what was changed). Handles script normalisation,
  sentencepiece tokenisation, sentence splitting/batching, and
  detokenisation/transliteration back to the target script.
- **`checkpoints/`** — the two int8 CTranslate2 model directories
  (`indic-en/`, `en-indic/`), ~214MB each. **Not committed to git** — see
  Setup above for how to fetch them.
- **`nmt_test.py`** — loads the translator, then does a Hindi ->
  English -> Hindi translation with timings, and exercises the fail-soft
  paths (same language, empty input, unsupported language). Run it to see
  translation quality before running the whole agent:

  ```bash
  python nmt/nmt_test.py
  ```

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
- **Local NMT translation is best-effort.** If the checkpoints are
  missing or fail to load (see Setup above), a startup warning disables
  translation and the agent still runs, just talking to the LLM in
  whatever language was transcribed. Any per-turn failure prints a
  warning and passes the text through untranslated so the turn still
  completes. Each turn costs two CPU inference passes; both show up as
  `Trans in:` / `Trans out:` in the timing summary. Typical latency with
  the int8 CTranslate2 checkpoints is well under 0.1s per direction for a
  few sentences on CPU.
- **The LLM always speaks English now.** `LLM_LANGUAGE` in
  `pipeline/main_optimized.py` is the middle language between the two
  local NMT stages, and `SYSTEM_PROMPT` tells Gemini to stay in English —
  the reply is translated back before TTS.
