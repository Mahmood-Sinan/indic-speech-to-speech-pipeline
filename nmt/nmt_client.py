"""Local Bhashini NMT stage (AI4Bharat IndicTrans2 via CTranslate2).

Ported from Suno Sutra's `bhashini_models` NMT server (see
`nmt/bhashini_engine/engine.py` for provenance) -- the same
int8-quantized IndicTrans2 CTranslate2 checkpoints Bhashini/Suno Sutra
run, wired in here as a direct Python import instead of the Flask
service + systemd setup Suno Sutra uses on-device.

Two directions are needed, because the LLM always works in English:

    user_language -> eng_Latn   (transcription, before the LLM)
    eng_Latn -> user_language   (LLM reply, before TTS)

Each direction is a separate ~214MB int8 CTranslate2 checkpoint, loaded
once at startup. Checkpoints live in `nmt/checkpoints/{indic-en,en-indic}/`
and are NOT committed to git (see README) -- regenerate them with:

    git -C suno-sutra-sw lfs pull \\
        --include="rootfs/roles/indic/files/bhashini_models.zip"

then extract `bhashini_models/nmt/checkpoints/{indic-en,en-indic}` from
that zip into `nmt/checkpoints/`.
"""

import os
import time

from nmt.bhashini_engine.engine import Model

# Flores-style language codes IndicTrans2 (and this engine) expects, keyed
# by the pipeline's short language codes (see LANGUAGE_NAMES in the
# pipeline). Same mapping the engine's own `flores_codes` table supports
# directly, so no separate ISO round-trip is needed.
#
# Three languages ship more than one script; the entries below pick the
# most widely used one, and are the only thing to edit if your STT model
# emits a different script:
#
#   ks  Kashmiri  -> kas_Arab (also available: kas_Deva)
#   mni Manipuri  -> mni_Beng (also available: mni_Mtei)
#   sd  Sindhi    -> snd_Arab (also available: snd_Deva)
INDIC_LANGUAGE_CODES = {
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "en": "eng_Latn",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "kok": "gom_Deva",
    "ks": "kas_Arab",
    "mai": "mai_Deva",
    "ml": "mal_Mlym",
    "mni": "mni_Beng",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "sa": "san_Deva",
    "sat": "sat_Olck",
    "sd": "snd_Arab",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}

CHECKPOINT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
INDIC_EN_CHECKPOINT = os.path.join(CHECKPOINT_ROOT, "indic-en", "ct2_int8_model")
EN_INDIC_CHECKPOINT = os.path.join(CHECKPOINT_ROOT, "en-indic", "ct2_int8_model")

DEFAULT_DEVICE = "cpu"
NUM_BEAMS = 5
MAX_DECODING_LENGTH = 256


class NMTError(RuntimeError):
    """A failure in the local NMT stage that callers can fail soft on."""


def _load_model(ckpt_dir, device, beam_size, max_decoding_length):
    if not os.path.isdir(ckpt_dir):
        raise NMTError(
            f"NMT checkpoint not found: {ckpt_dir}. Extract it from "
            "suno-sutra-sw's bhashini_models.zip (git lfs pull required) "
            "-- see nmt/nmt_client.py's module docstring."
        )
    return Model(
        ckpt_dir,
        device=device,
        input_lang_code_format="flores",
        model_type="ctranslate2",
        beam_size=beam_size,
        max_decoding_length=max_decoding_length,
    )


class IndicTransTranslator:
    """Translates between the user's language and English, locally."""

    def __init__(
        self,
        indic_en_checkpoint=INDIC_EN_CHECKPOINT,
        en_indic_checkpoint=EN_INDIC_CHECKPOINT,
        device=DEFAULT_DEVICE,
        num_beams=NUM_BEAMS,
        max_decoding_length=MAX_DECODING_LENGTH,
    ):
        self.device = device

        print(f"Loading local NMT (Bhashini/IndicTrans2 CTranslate2) models on {device}...")
        start = time.perf_counter()
        try:
            # Anything Indic -> English.
            self.indic_en = _load_model(
                indic_en_checkpoint, device, num_beams, max_decoding_length
            )
            # Anything English -> Indic.
            self.en_indic = _load_model(
                en_indic_checkpoint, device, num_beams, max_decoding_length
            )
        except NMTError:
            raise
        except Exception as exc:
            raise NMTError(f"could not load local NMT models: {exc}") from exc
        print(f"Local NMT models loaded in {time.perf_counter() - start:.1f}s.")

    @classmethod
    def from_env(cls):
        """Build a translator from NMT_* environment variables.

        Raises NMTError so callers can disable translation and keep going.
        """
        try:
            return cls(
                indic_en_checkpoint=os.environ.get(
                    "NMT_INDIC_EN_CHECKPOINT", INDIC_EN_CHECKPOINT
                ),
                en_indic_checkpoint=os.environ.get(
                    "NMT_EN_INDIC_CHECKPOINT", EN_INDIC_CHECKPOINT
                ),
                device=os.environ.get("NMT_DEVICE", DEFAULT_DEVICE),
            )
        except NMTError:
            raise
        except Exception as exc:
            raise NMTError(f"could not load local NMT models: {exc}") from exc

    def _direction(self, src_code, tgt_code):
        """Pick the (indic-en or en-indic) model that handles this direction."""
        if tgt_code == "eng_Latn":
            return self.indic_en
        if src_code == "eng_Latn":
            return self.en_indic
        raise NMTError(
            f"no model for {src_code}->{tgt_code}: only *->eng_Latn and "
            "eng_Latn->* are available"
        )

    def warm_up(self, source_language, target_language):
        """Run one tiny translation so failures surface at startup."""
        if source_language == target_language:
            return
        self.translate("hello" if source_language == "en" else "नमस्ते",
                       source_language, target_language)

    def translate(self, text, source_language, target_language):
        """Translate text, returning the result as a str.

        Raises NMTError for anything a caller should fail soft on; other
        exceptions are wrapped so callers only ever need one except clause.
        """
        if text is None:
            return text
        # Nothing to do: empty input, or same language on both ends.
        if not text.strip() or source_language == target_language:
            return text

        try:
            return self._translate(text, source_language, target_language)
        except NMTError:
            raise
        except Exception as exc:
            raise NMTError(
                f"{source_language}->{target_language} failed: {exc}"
            ) from exc

    def _translate(self, text, source_language, target_language):
        try:
            src_code = INDIC_LANGUAGE_CODES[source_language]
        except KeyError:
            raise NMTError(
                f"unsupported source language: {source_language}"
            ) from None
        try:
            tgt_code = INDIC_LANGUAGE_CODES[target_language]
        except KeyError:
            raise NMTError(
                f"unsupported target language: {target_language}"
            ) from None

        model = self._direction(src_code, tgt_code)

        # Sentence splitting, SPM tokenisation, batching and detokenisation
        # all happen inside the engine (see bhashini_engine/engine.py).
        return model.translate_paragraph(text, src_code, tgt_code)
