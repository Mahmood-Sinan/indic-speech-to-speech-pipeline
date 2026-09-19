import torch
import soundfile as sf
import sounddevice as sd

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


MODEL_ID = "ai4bharat/indic-parler-tts-pretrained"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
print("DEVICE:", DEVICE)

def play_audio(audio):
    # Play generated TTS output through the default playback device.

    sd.play(audio, model.config.sampling_rate)
    sd.wait() # wait till audio gets over


model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID
).to(DEVICE) # load model and move it to selected device(CPU or GPU)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID) 
# Tokenizer convert text to number so NN will understand. This is used to tokenize prompt

description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
) # This tokenizer is used to tokenize description of voice

# Text we want to speak
prompt = "ഹായ്, എനിക്ക് സുഖമാണ്. നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്? ഇന്ന് ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കും?"


# Description of the voice
description = (
    "A female Malayalam speaker speaks clearly and naturally "
    "with a friendly conversational tone at a moderate speed."
)

# Tokenize voice description
description_inputs = description_tokenizer(
    description,
    return_tensors="pt" # pt means pytorch tensors, the model expects so
)
# description tokenizer returns an object(can contains token ids, attention mask, etc...)
description_input_ids = description_inputs.input_ids.to(DEVICE)
# Extracts actual numbers(token ids) from output of description_tokenizer and moves into
# selected device.
description_attention_mask = (
    description_inputs.attention_mask.to(DEVICE)
)
# The attention mask tells the transformer which positions contain real tokens 
# and which positions are padding.

# Tokenize text to be spoken
prompt_inputs = tokenizer(
    prompt,
    return_tensors="pt",
)
prompt_input_ids = prompt_inputs.input_ids.to(DEVICE)
prompt_attention_mask = prompt_inputs.attention_mask.to(DEVICE)


print("Generating speech...")

with torch.inference_mode(): # Interface mode doesn't do grad calc, backpropagation, etc...
    generation = model.generate(
        input_ids=description_input_ids,
        attention_mask=description_attention_mask,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
    )
    # output of model which is the waveform represented as a pytorch tensor

# Convert to NumPy
audio = generation.cpu().numpy().squeeze()
# .cpu() -> move the tensor to cpu, because after this we use np and sf which doesn't require gpu
# .numpy() -> because sf expects a np array
# .squeeze() -> if audio is [1,n] or [n,1] squeeze makes them [n] (removes dimension of size 1).


# Save WAV
output_file = "tts_output.wav"

sf.write(
    output_file,
    audio,
    model.config.sampling_rate,
)
play_audio(audio)

print(f"Saved: {output_file}")
print(f"Sample rate: {model.config.sampling_rate} Hz")
print(f"Audio samples: {len(audio)}")
print(f"Duration: {len(audio) / model.config.sampling_rate:.2f} seconds")
