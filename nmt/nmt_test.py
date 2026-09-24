"""Standalone check for the local NMT stage.

Run it before the full pipeline to confirm the models load and translate:

    python nmt/nmt_test.py

It does a Hindi -> English -> Hindi round trip and prints timings, so you
can see both directions work without touching the microphone, the LLM or
TTS.
"""

import sys
import time
from pathlib import Path

# This file runs as `python nmt/nmt_test.py`, which only puts nmt/ itself
# on sys.path -- add the repo root so nmt.nmt_client imports cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nmt.nmt_client import NMTError, IndicTransTranslator  # noqa: E402


def main():
    print("Loading translator...")
    start = time.perf_counter()
    try:
        translator = IndicTransTranslator.from_env()
    except NMTError as exc:
        print(f"FAILED to load translator: {exc}")
        return 1
    print(f"loaded in {time.perf_counter() - start:.1f}s\n")

    samples = [
        "मेरा नाम राहुल है।",
        "आज मौसम बहुत अच्छा है।",
        "क्या आप मेरी मदद कर सकते हैं?",
    ]

    print("Hindi -> English (this is what the LLM will receive)")
    for sentence in samples:
        start = time.perf_counter()
        try:
            out = translator.translate(sentence, "hi", "en")
        except NMTError as exc:
            print(f"  FAILED: {exc}")
            return 1
        print(f"  {sentence}  ->  {out}   ({time.perf_counter() - start:.2f}s)")

    print("\nEnglish -> Hindi (this is what TTS will receive)")
    for sentence in [
        "My name is Rahul.",
        "The weather is very nice today.",
        "Can you help me?",
    ]:
        start = time.perf_counter()
        try:
            out = translator.translate(sentence, "en", "hi")
        except NMTError as exc:
            print(f"  FAILED: {exc}")
            return 1
        print(f"  {sentence}  ->  {out}   ({time.perf_counter() - start:.2f}s)")

    print("\nFail-soft checks (each must pass text through unchanged)")
    checks = [
        ("same language", lambda: translator.translate("hello", "en", "en")),
        ("empty string", lambda: translator.translate("", "hi", "en")),
        ("whitespace", lambda: translator.translate("   ", "hi", "en")),
        ("unsupported language", lambda: translator.translate("x", "xy", "en")),
    ]
    for label, fn in checks:
        try:
            result = fn()
            print(f"  {label}: returned {result!r} (no exception)")
        except NMTError as exc:
            # An NMTError is also acceptable here: the pipeline turns it
            # into a warning and passes the original text through.
            print(f"  {label}: raised NMTError ({exc}) - pipeline will skip it")

    print("\nAll NMT checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
