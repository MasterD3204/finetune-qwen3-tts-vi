# coding=utf-8
# Vietnamese data preparation for Qwen3-TTS fine-tuning
#
# Extends prepare_data.py to support:
#   1. Mixed data (Vietnamese + original language) with configurable ratio
#   2. Audio validation (duration, sample rate, SNR check)
#   3. Vietnamese text normalization (numbers, abbreviations, tone marks)
#   4. Data shuffling and splitting (train/val)
#   5. Manifest generation for monitoring
#
# Usage:
#   # Vietnamese-only dataset
#   python prepare_data_vi.py \
#     --device cuda:0 \
#     --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
#     --vi_jsonl ./data/vi_speaker_raw.jsonl \
#     --output_jsonl ./data/train_vi_with_codes.jsonl
#
#   # Mixed dataset (Vietnamese + original)
#   python prepare_data_vi.py \
#     --device cuda:0 \
#     --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
#     --vi_jsonl ./data/vi_general_raw.jsonl \
#     --original_jsonl ./data/original_raw.jsonl \
#     --vi_ratio 0.8 \
#     --output_jsonl ./data/train_mixed_with_codes.jsonl

import argparse
import json
import os
import random
import re
import warnings
from typing import List, Optional, Tuple

import librosa
import numpy as np

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32

# ---------------------------------------------------------------------------
# Vietnamese text normalization
# ---------------------------------------------------------------------------

# Map Vietnamese tone marks to IPA-like representation for debugging
_TONE_MAP = {
    "\u0300": "huyền",  # grave
    "\u0301": "sắc",    # acute
    "\u0303": "ngã",    # tilde
    "\u0309": "hỏi",    # hook above
    "\u0323": "nặng",   # dot below
}

# Vietnamese abbreviations → full form
_VI_ABBREV = {
    "tp.": "thành phố",
    "tp ": "thành phố ",
    "q.": "quận",
    "h.": "huyện",
    "vnd": "đồng",
    "vnđ": "đồng",
    "đ": "đồng",
    "km": "ki lô mét",
    "cm": "xăng ti mét",
    "m²": "mét vuông",
    "m2": "mét vuông",
    "kg": "ki lô gam",
    "g ": "gam ",
    "mg": "mi li gam",
    "ml": "mi li lít",
    "l ": "lít ",
    "&": "và",
    "%": " phần trăm",
}

# Vietnamese number words
_VI_ONES = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_VI_TEENS = ["mười", "mười một", "mười hai", "mười ba", "mười bốn", "mười lăm",
             "mười sáu", "mười bảy", "mười tám", "mười chín"]


def _number_to_vi_words(n: int) -> str:
    """Convert integer to Vietnamese spoken form (simplified, handles 0–999999)."""
    if n < 0:
        return "âm " + _number_to_vi_words(-n)
    if n == 0:
        return "không"
    if n < 10:
        return _VI_ONES[n]
    if n < 20:
        return _VI_TEENS[n - 10]
    if n < 100:
        tens = n // 10
        ones = n % 10
        result = _VI_ONES[tens] + " mươi"
        if ones == 1:
            result += " mốt"
        elif ones == 5:
            result += " lăm"
        elif ones > 0:
            result += " " + _VI_ONES[ones]
        return result
    if n < 1000:
        hundreds = n // 100
        remainder = n % 100
        result = _VI_ONES[hundreds] + " trăm"
        if remainder > 0:
            if remainder < 10:
                result += " lẻ " + _VI_ONES[remainder]
            else:
                result += " " + _number_to_vi_words(remainder)
        return result
    if n < 1_000_000:
        thousands = n // 1000
        remainder = n % 1000
        result = _number_to_vi_words(thousands) + " nghìn"
        if remainder > 0:
            result += " " + _number_to_vi_words(remainder)
        return result
    # Fallback: digit by digit
    return " ".join(_VI_ONES[int(d)] for d in str(n))


def _replace_numbers(text: str) -> str:
    """Replace Arabic numerals with Vietnamese spoken form."""
    def replace_match(m):
        num_str = m.group(0).replace(",", "").replace(".", "")
        try:
            return _number_to_vi_words(int(num_str))
        except ValueError:
            return m.group(0)

    # Match numbers with optional thousand separators
    return re.sub(r"\d{1,3}(?:[,\.]\d{3})*|\d+", replace_match, text)


def normalize_vietnamese_text(text: str) -> str:
    """
    Light normalization for Vietnamese TTS input text:
    - Lowercase abbreviations expansion
    - Number → words conversion
    - Remove/replace unusual punctuation
    - Normalize whitespace
    """
    # Expand abbreviations (case-insensitive)
    lower = text.lower()
    for abbrev, expansion in _VI_ABBREV.items():
        lower = lower.replace(abbrev, expansion)
    text = lower

    # Convert numbers to words
    text = _replace_numbers(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove characters outside Vietnamese + common punctuation
    # Keep: Vietnamese unicode block, ASCII letters, common punctuation
    text = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF,\.!\?;:\-\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Audio validation
# ---------------------------------------------------------------------------

MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0
TARGET_SR = 24000


def validate_audio(audio_path: str) -> Tuple[bool, str]:
    """
    Check if audio file is suitable for TTS training.
    Returns (ok: bool, reason: str).
    """
    if not os.path.exists(audio_path):
        return False, f"File not found: {audio_path}"

    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        return False, f"Load error: {e}"

    duration = len(audio) / sr
    if duration < MIN_DURATION_S:
        return False, f"Too short: {duration:.2f}s < {MIN_DURATION_S}s"
    if duration > MAX_DURATION_S:
        return False, f"Too long: {duration:.2f}s > {MAX_DURATION_S}s"

    # Basic energy check (silence detection)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-4:
        return False, f"Too quiet: RMS={rms:.6f}"

    return True, "ok"


def validate_and_filter(
    data_list: List[dict],
    validate: bool = True,
    normalize_text: bool = True,
    language: str = "vi",
) -> Tuple[List[dict], List[dict]]:
    """
    Filter and normalize a list of data dicts.
    Returns (valid_list, rejected_list).
    """
    valid = []
    rejected = []

    for item in data_list:
        # Ensure language field
        item.setdefault("language", language)

        # Validate audio
        if validate:
            ok, reason = validate_audio(item.get("audio", ""))
            if not ok:
                item["_reject_reason"] = reason
                rejected.append(item)
                continue

            if "ref_audio" in item:
                ok_ref, reason_ref = validate_audio(item["ref_audio"])
                if not ok_ref:
                    item["_reject_reason"] = f"ref_audio: {reason_ref}"
                    rejected.append(item)
                    continue

        # Normalize text
        if normalize_text and item.get("language") in ("vi", "Vietnamese"):
            item["text"] = normalize_vietnamese_text(item["text"])

        valid.append(item)

    return valid, rejected


# ---------------------------------------------------------------------------
# Mixed dataset builder
# ---------------------------------------------------------------------------

def build_mixed_dataset(
    vi_data: List[dict],
    original_data: Optional[List[dict]],
    vi_ratio: float = 0.8,
    max_total: int = 0,
    seed: int = 42,
) -> List[dict]:
    """
    Build a mixed dataset with vi_ratio fraction of Vietnamese data.
    If original_data is None, returns only Vietnamese data.

    Args:
        vi_data: Vietnamese utterances.
        original_data: Original language utterances (can be None).
        vi_ratio: Fraction of Vietnamese data in mixed dataset.
        max_total: If > 0, cap total dataset size.
        seed: Random seed for reproducibility.

    Returns:
        Shuffled mixed list.
    """
    rng = random.Random(seed)

    if original_data is None or len(original_data) == 0:
        result = list(vi_data)
    else:
        n_total = len(vi_data) + len(original_data)
        if max_total > 0:
            n_total = min(n_total, max_total)

        n_vi = min(len(vi_data), int(n_total * vi_ratio))
        n_orig = min(len(original_data), n_total - n_vi)

        vi_sample = rng.sample(vi_data, n_vi) if n_vi < len(vi_data) else list(vi_data)
        orig_sample = rng.sample(original_data, n_orig) if n_orig < len(original_data) else list(original_data)

        result = vi_sample + orig_sample
        print(f"[data] Mixed: {n_vi} Vietnamese + {n_orig} original = {len(result)} total")

    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Audio tokenization (same as original prepare_data.py)
# ---------------------------------------------------------------------------

def tokenize_audio_codes(
    tokenizer: "Qwen3TTSTokenizer",
    data_list: List[dict],
    batch_size: int = BATCH_INFER_NUM,
) -> List[dict]:
    """
    Add 'audio_codes' field to each item by tokenizing audio files.
    Items that fail to tokenize are skipped with a warning.
    """
    final = []
    batch_items = []
    batch_paths = []

    def flush_batch():
        if not batch_paths:
            return
        try:
            enc_res = tokenizer.encode(batch_paths)
            for code, item in zip(enc_res.audio_codes, batch_items):
                item["audio_codes"] = code.cpu().tolist()
                final.append(item)
        except Exception as e:
            warnings.warn(f"[tokenize] Batch failed: {e}. Skipping {len(batch_paths)} items.")
        batch_items.clear()
        batch_paths.clear()

    for item in data_list:
        batch_items.append(item)
        batch_paths.append(item["audio"])
        if len(batch_items) >= batch_size:
            flush_batch()

    flush_batch()
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare Vietnamese TTS training data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str,
                        default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--vi_jsonl", type=str, required=True,
                        help="Raw Vietnamese JSONL: {audio, text, ref_audio}")
    parser.add_argument("--original_jsonl", type=str, default=None,
                        help="(Optional) Original language raw JSONL for data mixing")
    parser.add_argument("--vi_ratio", type=float, default=0.8,
                        help="Fraction of Vietnamese data in mixed set (default: 0.8)")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Output path for train_with_codes.jsonl")
    parser.add_argument("--output_val_jsonl", type=str, default=None,
                        help="(Optional) Output path for validation split")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Fraction to hold out for validation (default: 0.05 = 5%%)")
    parser.add_argument("--max_total", type=int, default=0,
                        help="Cap total dataset size (0 = no cap)")
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip audio validation (faster but risky)")
    parser.add_argument("--no_normalize_text", action="store_true",
                        help="Skip Vietnamese text normalization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=BATCH_INFER_NUM)
    parser.add_argument("--rejected_jsonl", type=str, default="rejected.jsonl",
                        help="Where to write rejected audio entries")
    args = parser.parse_args()

    random.seed(args.seed)

    # -----------------------------------------------------------------------
    # Load Vietnamese data
    # -----------------------------------------------------------------------
    print(f"[data] Loading Vietnamese data from {args.vi_jsonl}...")
    with open(args.vi_jsonl, encoding="utf-8") as f:
        vi_raw = [json.loads(l) for l in f if l.strip()]
    print(f"[data] Loaded {len(vi_raw)} Vietnamese entries")

    vi_valid, vi_rejected = validate_and_filter(
        vi_raw,
        validate=not args.no_validate,
        normalize_text=not args.no_normalize_text,
        language="vi",
    )
    print(f"[data] Vietnamese: {len(vi_valid)} valid, {len(vi_rejected)} rejected")

    # -----------------------------------------------------------------------
    # Load original language data (optional)
    # -----------------------------------------------------------------------
    original_valid = None
    if args.original_jsonl:
        print(f"[data] Loading original language data from {args.original_jsonl}...")
        with open(args.original_jsonl, encoding="utf-8") as f:
            orig_raw = [json.loads(l) for l in f if l.strip()]
        original_valid, orig_rejected = validate_and_filter(
            orig_raw,
            validate=not args.no_validate,
            normalize_text=False,
        )
        print(f"[data] Original: {len(original_valid)} valid, {len(orig_rejected)} rejected")
        vi_rejected.extend(orig_rejected)

    # -----------------------------------------------------------------------
    # Save rejected entries
    # -----------------------------------------------------------------------
    if vi_rejected:
        with open(args.rejected_jsonl, "w", encoding="utf-8") as f:
            for item in vi_rejected:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[data] {len(vi_rejected)} rejected entries written to {args.rejected_jsonl}")

    # -----------------------------------------------------------------------
    # Build mixed dataset
    # -----------------------------------------------------------------------
    mixed = build_mixed_dataset(
        vi_valid,
        original_valid,
        vi_ratio=args.vi_ratio,
        max_total=args.max_total,
        seed=args.seed,
    )
    print(f"[data] Final mixed dataset: {len(mixed)} entries")

    # -----------------------------------------------------------------------
    # Train / val split
    # -----------------------------------------------------------------------
    if args.output_val_jsonl and args.val_ratio > 0:
        rng = random.Random(args.seed)
        rng.shuffle(mixed)
        n_val = max(1, int(len(mixed) * args.val_ratio))
        val_data = mixed[:n_val]
        train_data = mixed[n_val:]
        print(f"[data] Split: {len(train_data)} train / {len(val_data)} val")
    else:
        train_data = mixed
        val_data = []

    # -----------------------------------------------------------------------
    # Tokenize audio → audio_codes
    # -----------------------------------------------------------------------
    print(f"[tokenize] Loading speech tokenizer from {args.tokenizer_model_path}...")
    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    print(f"[tokenize] Encoding {len(train_data)} training utterances...")
    train_tokenized = tokenize_audio_codes(tokenizer_12hz, train_data, args.batch_size)
    print(f"[tokenize] Done: {len(train_tokenized)} training utterances tokenized")

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in train_tokenized:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[output] Training data written to {args.output_jsonl}")

    if val_data and args.output_val_jsonl:
        print(f"[tokenize] Encoding {len(val_data)} validation utterances...")
        val_tokenized = tokenize_audio_codes(tokenizer_12hz, val_data, args.batch_size)
        with open(args.output_val_jsonl, "w", encoding="utf-8") as f:
            for item in val_tokenized:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[output] Validation data written to {args.output_val_jsonl}")

    # -----------------------------------------------------------------------
    # Statistics summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Data Preparation Summary")
    print("=" * 50)
    print(f"  Vietnamese entries (raw):  {len(vi_raw)}")
    print(f"  Vietnamese entries (valid): {len(vi_valid)}")
    if original_valid is not None:
        print(f"  Original entries (valid):  {len(original_valid)}")
    print(f"  Total after mixing:        {len(mixed)}")
    print(f"  Training set:              {len(train_tokenized)}")
    if val_data:
        print(f"  Validation set:            {len(val_data)}")
    print(f"  Rejected entries:          {len(vi_rejected)}")

    # Show sample
    if train_tokenized:
        sample = train_tokenized[0]
        print(f"\n  Sample entry:")
        print(f"    audio: {sample.get('audio', '')}")
        print(f"    text:  {sample.get('text', '')[:80]}")
        print(f"    language: {sample.get('language', '')}")
        codes = sample.get("audio_codes", [])
        print(f"    audio_codes: {len(codes)} frames × {len(codes[0]) if codes else 0} codebooks")
    print("=" * 50)


if __name__ == "__main__":
    main()
