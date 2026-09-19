import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import queue
import time

import numpy as np
import sounddevice as sd
import torch

from silero_vad import load_silero_vad
from transformers import AutoModel, AutoTokenizer

from google import genai
from parler_tts import ParlerTTSForConditionalGeneration


SAMPLE_RATE = 16000

# Your working ALSA device:
DEVICE = 4

# Hardware gives us 2 channels
CHANNELS = 2

# Number of samples to pass to VAD Silero to detect voice
CHUNK_SIZE = 512

# VAD probability threshold
VAD_THRESHOLD = 0.5

# Require several consecutive speech chunks.
SPEECH_CHUNKS_REQUIRED = 3

# Require several consecutive silent chunks to end utterance
# Length of silence to declare that the speech has ended.
SILENCE_CHUNKS_REQUIRED = 10

# Language you will speak
#
# hi = Hindi
# ta = Tamil
# te = Telugu
# kn = Kannada
# ml = Malayalam
# bn = Bengali
LANGUAGE = "ml"

# CTC for the first integrated test
DECODING = "ctc"

STT_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

LLM_MODEL_ID = "gemini-3.5-flash-lite"

TTS_MODEL_ID = "ai4bharat/indic-parler-tts-pretrained"

# Voice description.
# controls the characteristics of the generated voice.
TTS_DESCRIPTION = (
    "A female Malayalam speaker speaks clearly and naturally "
    "with a friendly conversational tone at a moderate speed."
)


# ============================================================
# Device configuration
# ============================================================

STT_DEVICE = "cpu"
TTS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAD_DEVICE = "cpu"
# Change to TTS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu" if the GPU has enough VRAM

print("=" * 60)
print("Indic Voice Agent")
print("=" * 60)

print("PyTorch device:")
print("  STT:", STT_DEVICE)
print("  TTS:", TTS_DEVICE)
print("  VAD:", VAD_DEVICE)

print()


# ============================================================
# Load VAD
# ============================================================

print("Loading Silero VAD...")

vad_model = load_silero_vad()

vad_model = vad_model.to(VAD_DEVICE) # Move VAD to selected Device for VAD
vad_model.eval() # puts VAD into evaluation mode (not training mode)

print("Silero VAD loaded.")


# ============================================================
# Load STT
# ============================================================

print("\nLoading IndicConformer...")

# IndicConformer's remote code (model_onnx.py) picks its device with its own
# `torch.cuda.is_available()` check at construction time -- it ignores the
# `.to(STT_DEVICE)` call below entirely, since its ~27 onnxruntime
# InferenceSessions aren't registered as torch submodules. Left alone, it
# silently loads every ONNX session (encoder + per-language joint nets) onto
# CUDA via its own memory arena, invisible to torch's allocator, which is
# what was actually exhausting the 6GB card. Fake CUDA's absence for just
# this constructor call so it honors STT_DEVICE = "cpu".
_real_cuda_is_available = torch.cuda.is_available
if STT_DEVICE == "cpu":
    torch.cuda.is_available = lambda: False

try:
    stt_model = AutoModel.from_pretrained(
        STT_MODEL_ID,
        trust_remote_code=True,
    ).to(STT_DEVICE)
finally:
    torch.cuda.is_available = _real_cuda_is_available

stt_model.eval() # puts stt model into evaluation mode (not training mode)

print("IndicConformer loaded.")
print("STT model device:", stt_model.d)


# ============================================================
# Load Gemini
# ============================================================

print("\nInitializing Gemini...")

llm_client = genai.Client()

print("Gemini client initialized.")


def ask_llm(text):
    response = llm_client.interactions.create(
        model=LLM_MODEL_ID,
        input=text
    )
    # asks the text prompt to gemini and return the reponse 
    return response.output_text


# ============================================================
# Load TTS
# ============================================================

print("\nLoading Indic Parler-TTS...")

tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
    TTS_MODEL_ID,
    torch_dtype=torch.float16,
).to(TTS_DEVICE) # Move TTS model to selected Device for TTS model

tts_model.eval() # puts TTS model into evaluation mode (not training mode)

tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
# Tokenizer convert text to number so NN will understand. This is used to tokenize the text prompt

tts_description_tokenizer = AutoTokenizer.from_pretrained(
    tts_model.config.text_encoder._name_or_path
) # This tokenizer is used to tokenize description of voice

TTS_SAMPLE_RATE = tts_model.config.sampling_rate #1

print("Indic Parler-TTS loaded.")
print("TTS device:", TTS_DEVICE)
print("TTS sample rate:", TTS_SAMPLE_RATE) #1


# ============================================================
# Call TTS
# ============================================================


def call_tts(text):
    # Convert text into a waveform using TTS model.

    # Returns: NumPy array containing the generated waveform.

    # --------------------------------------------------------
    # Tokenize voice description
    # --------------------------------------------------------
    print("text: ", text)
    description_inputs = tts_description_tokenizer(
        TTS_DESCRIPTION,
        return_tensors="pt", # pt means pytorch tensors, the model expects so
    )

    # description tokenizer returns an object(can contains token ids, attention mask, etc...)
    description_input_ids = description_inputs.input_ids.to(TTS_DEVICE)
    # Extracts actual numbers(token ids) from output of description_tokenizer and moves into
    # selected device.
    description_attention_mask = description_inputs.attention_mask.to(TTS_DEVICE)
    # The attention mask tells the transformer which positions contain real tokens 
    # and which positions are padding.

    # --------------------------------------------------------
    # Tokenize text to speak
    # --------------------------------------------------------

    prompt_inputs = tts_tokenizer(
        text,
        return_tensors="pt",
    )

    prompt_input_ids = prompt_inputs.input_ids.to(TTS_DEVICE)

    prompt_attention_mask = prompt_inputs.attention_mask.to(TTS_DEVICE)

    # --------------------------------------------------------
    # Generate speech
    # --------------------------------------------------------
    # if TTS_DEVICE == "cuda":
    #     torch.cuda.synchronize()

    # start_time = time.perf_counter()

    with torch.inference_mode(): # Interface mode doesn't do grad calc, backpropagation, etc...
        generation = tts_model.generate(
            input_ids=description_input_ids,
            attention_mask=description_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )
    # output of model which is the waveform represented as a pytorch tensor

    # if TTS_DEVICE == "cuda":
    #     torch.cuda.synchronize()

    # generation_time = time.perf_counter() - start_time

    audio = generation.cpu().numpy().squeeze().astype(np.float32)
    # .cpu() -> move the tensor to cpu, because after this we use np and sf which doesn't require gpu
    # .numpy() -> because sf expects a np array
    # .squeeze() -> if audio is [1,n] or [n,1] squeeze makes them [n] (removes dimension of size 1).
    # .astype(np.float32) -> the model runs in fp16, but sounddevice.play() only
    # supports float32/int32/int16/int8/uint8, not float16.

    # --------------------------------------------------------
    # Free GPU memory used by this generation
    # --------------------------------------------------------
    # Every turn allocates new description/prompt/KV-cache tensors of a
    # different shape. Without releasing them, PyTorch's CUDA allocator
    # fragments across turns until the 6GB card has no free block left,
    # even though the model itself only needs ~2GB.
    del generation, description_inputs, description_input_ids
    del description_attention_mask, prompt_inputs, prompt_input_ids
    del prompt_attention_mask

    if TTS_DEVICE == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Calculate audio duration / RTF
    # --------------------------------------------------------

    # audio_duration = len(audio) / TTS_SAMPLE_RATE

    # rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")

    # print(
    #     f"TTS generation: {generation_time:.2f}s | "
    #     f"audio: {audio_duration:.2f}s | "
    #     f"RTF: {rtf:.3f}"
    # )

    return audio


# ============================================================
# Audio playback
# ============================================================


def play_audio(audio, tts_model_sampling_rate):
    # Play generated TTS output through the default playback device.

    sd.play(audio, tts_model_sampling_rate)
    sd.wait() # wait till audio gets over


# ============================================================
# Audio queue
# ============================================================

audio_queue = queue.Queue(maxsize=100)


def audio_callback(
    indata,
    frames,
    time_info,
    status,
):
    """
    Called by sounddevice whenever a new
    microphone chunk arrives.
    """

    if status:
        print("Audio status:", status)

    try:

        audio_queue.put_nowait(indata.copy())

    except queue.Full:

        print("WARNING: Audio queue is full. " "Dropping audio chunk.")


# ============================================================
# Clear queued microphone audio
# ============================================================


def clear_audio_queue():
    """
    Remove microphone audio that accumulated while
    the assistant was processing / speaking.

    This prevents old audio from being interpreted as
    a new user utterance after TTS finishes.
    """

    cleared = 0

    while True:

        try:

            audio_queue.get_nowait()

            cleared += 1

        except queue.Empty:

            break

    if cleared > 0:

        print(f"Cleared {cleared} queued " f"microphone chunks.")


# ============================================================
# Start microphone
# ============================================================

print("\nStarting microphone...")

print("Speak in:", LANGUAGE)

print("Press Ctrl+C to stop.")

print()


# Tells sounddevice to open the microphone and continuously provide audio to audio_callback()
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=CHUNK_SIZE,
    channels=CHANNELS,
    dtype="float32",
    device=DEVICE,
    callback=audio_callback,
)
stream.start()


# ============================================================
# State
# ============================================================

speech_active = False

speech_count = 0
silence_count = 0

utterance_chunks = []


# ============================================================
# Main loop
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Get next microphone chunk
        # ----------------------------------------------------

        chunk = audio_queue.get()

        # chunk shape:
        #
        # (512, 2)
        #
        # 512 samples
        # 2 channels

        # ----------------------------------------------------
        # Stereo -> mono
        # ----------------------------------------------------

        mono = chunk.mean(axis=1)

        # ----------------------------------------------------
        # NumPy -> PyTorch
        # ----------------------------------------------------

        vad_audio = torch.from_numpy(mono).float().to(VAD_DEVICE)

        # ----------------------------------------------------
        # Run VAD
        # ----------------------------------------------------

        with torch.inference_mode():

            speech_probability = vad_model(vad_audio, SAMPLE_RATE).item()

        is_speech = speech_probability >= VAD_THRESHOLD

        # ----------------------------------------------------
        # Update VAD counters
        # ----------------------------------------------------

        if is_speech:

            speech_count += 1
            silence_count = 0

        else:

            silence_count += 1
            speech_count = 0

        # ====================================================
        # Detect SPEECH START
        # ====================================================

        if not speech_active:

            if speech_count >= SPEECH_CHUNKS_REQUIRED:

                speech_active = True

                # Start with current chunk
                utterance_chunks = [chunk]

                silence_count = 0

            continue

        # ====================================================
        # Currently inside an utterance
        # ====================================================

        utterance_chunks.append(chunk)

        # ====================================================
        # Detect SPEECH END
        # ====================================================

        if silence_count >= SILENCE_CHUNKS_REQUIRED:

            speech_active = False

            # ------------------------------------------------
            # Combine chunks
            # ------------------------------------------------

            utterance = np.concatenate(
                utterance_chunks,
                axis=0,
            )

            # ------------------------------------------------
            # Stereo -> mono
            # ------------------------------------------------

            utterance = utterance.mean(axis=1)

            duration = len(utterance) / SAMPLE_RATE

            print()
            print(f"Utterance detected " f"({duration:.2f}s)")

            # ------------------------------------------------
            # Convert to PyTorch
            # ------------------------------------------------

            wav = torch.from_numpy(utterance).unsqueeze(0).to(stt_model.d)

            # =================================================
            # STT
            # =================================================

            print("Transcribing...")

            stt_start = time.perf_counter()

            with torch.inference_mode():

                transcription = stt_model(
                    wav,
                    LANGUAGE,
                    DECODING,
                )

            if STT_DEVICE == "cuda":

                torch.cuda.synchronize()

            stt_time = time.perf_counter() - stt_start

            print(f"You: {transcription}")

            print(f"STT time: " f"{stt_time:.2f}s")

            # =================================================
            # LLM
            # =================================================

            print("Thinking...")

            llm_start = time.perf_counter()

            llm_response = ask_llm(transcription)

            llm_time = time.perf_counter() - llm_start

            print(f"Gemini: {llm_response}")

            print(f"LLM time: " f"{llm_time:.2f}s")

            # =================================================
            # TTS
            # =================================================

            print("Generating speech...")

            tts_audio = call_tts(llm_response)
            tts_model_sampling_rate = tts_model.config.sampling_rate
            # =================================================
            # Playback
            # =================================================

            print("Playing response...")

            playback_start = time.perf_counter()

            play_audio(tts_audio, tts_model_sampling_rate)

            playback_time = time.perf_counter() - playback_start

            # =================================================
            # Total processing information
            # =================================================

            print()
            print("========== TIMING ==========")

            print(f"STT:       {stt_time:.2f}s")

            print(f"Gemini:    {llm_time:.2f}s")

            print(f"TTS + gen: see above")

            print(f"Playback:  {playback_time:.2f}s")

            print("============================")

            print()

            # ------------------------------------------------
            # Clear microphone audio captured while
            # processing / speaking
            # ------------------------------------------------

            clear_audio_queue()

            # ------------------------------------------------
            # Reset utterance state
            # ------------------------------------------------

            utterance_chunks = []

            silence_count = 0

            speech_count = 0


except KeyboardInterrupt:

    print("\nStopping...")


finally:

    stream.stop()
    stream.close()

    print("Microphone stopped.")
