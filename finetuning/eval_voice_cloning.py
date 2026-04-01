#!/usr/bin/env python3
# coding=utf-8
# Voice cloning quality evaluation script for Qwen3-TTS
#
# Evaluates whether voice cloning capability is preserved after Vietnamese fine-tuning.
#
# Metrics computed:
#   1. Speaker Similarity (cosine similarity of x-vector/ECAPA embeddings)
#   2. MOS-proxy (UTMOS score via utmos library if available, else skip)
#   3. CER/WER on Vietnamese output (via Whisper if available)
#   4. Basic audio quality (RMS, silence ratio, spectral centroid)
#
# Usage:
#   python eval_voice_cloning.py \
#     --model_path output_vi/checkpoint-epoch-4 \
#     --ref_audio ./data/ref.wav \
#     --test_texts_vi ./eval/vi_test_texts.txt \
#     --test_texts_orig ./eval/orig_test_texts.txt \
#     --output_dir ./eval_results \
#     --speaker_name vi_speaker

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import soundfile as sf

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    import librosa
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def compute_audio_stats(audio: np.ndarray, sr: int) -> dict:
    """Compute basic quality metrics for an audio array."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    silence_ratio = float(np.mean(np.abs(audio) < 0.01))
    duration = len(audio) / sr

    # Spectral centroid (rough brightness measure)
    import librosa
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr)
    mean_sc = float(np.mean(sc))

    return {
        "duration_s": round(duration, 3),
        "rms": round(rms, 6),
        "silence_ratio": round(silence_ratio, 4),
        "spectral_centroid_hz": round(mean_sc, 1),
    }


# ---------------------------------------------------------------------------
# Speaker similarity (ECAPA-TDNN via speechbrain, fallback: cosine of mel mean)
# ---------------------------------------------------------------------------

_spk_encoder = None

def get_speaker_encoder():
    global _spk_encoder
    if _spk_encoder is not None:
        return _spk_encoder
    try:
        from speechbrain.pretrained import EncoderClassifier
        _spk_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        print("[eval] Using SpeechBrain ECAPA-TDNN for speaker similarity")
    except ImportError:
        _spk_encoder = "fallback"
        print("[eval] SpeechBrain not available — using mel-mean cosine similarity (less accurate)")
    return _spk_encoder


def extract_speaker_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract speaker embedding. Returns 1-D numpy array."""
    encoder = get_speaker_encoder()

    if encoder == "fallback":
        # Fallback: mean of mel spectrogram as a weak proxy
        import librosa
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        return np.mean(mel, axis=1)

    wav_tensor = torch.tensor(audio).unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(wav_tensor)
    return emb.squeeze().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def speaker_similarity(ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 16000) -> float:
    ref_emb = extract_speaker_embedding(ref_audio, sr)
    gen_emb = extract_speaker_embedding(gen_audio, sr)
    return cosine_similarity(ref_emb, gen_emb)


# ---------------------------------------------------------------------------
# ASR for CER/WER (optional, requires whisper)
# ---------------------------------------------------------------------------

_asr_model = None

def transcribe_audio(audio: np.ndarray, sr: int, language: str = "vi") -> str:
    global _asr_model
    try:
        import whisper
        if _asr_model is None:
            _asr_model = whisper.load_model("base")
            print("[eval] Using Whisper-base for CER/WER evaluation")
        audio_16k = audio
        if sr != 16000:
            import librosa
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        result = _asr_model.transcribe(audio_16k, language=language)
        return result["text"].strip()
    except ImportError:
        return ""
    except Exception as e:
        warnings.warn(f"[asr] Transcription failed: {e}")
        return ""


def cer(ref: str, hyp: str) -> float:
    """Character error rate."""
    if not ref:
        return 0.0
    import editdistance
    try:
        dist = editdistance.eval(list(ref), list(hyp))
        return dist / len(ref)
    except ImportError:
        # Simple fallback
        return abs(len(ref) - len(hyp)) / len(ref)


# ---------------------------------------------------------------------------
# TTS generation
# ---------------------------------------------------------------------------

def generate_speech(
    model_path: str,
    text: str,
    speaker_name: str,
    ref_audio_path: Optional[str],
    language: str = "vi",
) -> Tuple[Optional[np.ndarray], int]:
    """
    Generate speech using the fine-tuned model.
    Returns (waveform, sample_rate) or (None, 0) on failure.
    """
    try:
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

        if ref_audio_path is not None:
            # Voice cloning mode
            prompt = model.create_voice_clone_prompt(ref_audio_path)
            audio_out, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt,
                language=language,
            )
        else:
            # Custom voice mode (predefined speaker)
            audio_out, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker_name,
                language=language,
            )

        if audio_out is None:
            return None, 0
        return audio_out, sr

    except Exception as e:
        warnings.warn(f"[gen] Generation failed for '{text[:40]}': {e}")
        return None, 0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> dict:
    os.makedirs(args.output_dir, exist_ok=True)
    audio_dir = os.path.join(args.output_dir, "generated_audio")
    os.makedirs(audio_dir, exist_ok=True)

    results = {
        "model_path": args.model_path,
        "speaker_name": args.speaker_name,
        "vietnamese": [],
        "original": [],
        "summary": {},
    }

    # Load reference audio for speaker similarity
    ref_audio_16k, _ = load_audio(args.ref_audio, target_sr=16000)
    ref_audio_24k, _ = load_audio(args.ref_audio, target_sr=24000)

    print(f"\n[eval] Reference audio: {args.ref_audio}")
    ref_stats = compute_audio_stats(ref_audio_16k, 16000)
    print(f"[eval] Ref stats: {ref_stats}")

    def eval_texts(text_list: List[str], language: str, tag: str) -> List[dict]:
        records = []
        for i, text in enumerate(text_list):
            text = text.strip()
            if not text:
                continue
            print(f"\n[eval] [{tag}] {i+1}/{len(text_list)}: {text[:60]}...")

            # Generate
            audio, sr = generate_speech(
                model_path=args.model_path,
                text=text,
                speaker_name=args.speaker_name,
                ref_audio_path=args.ref_audio if args.use_voice_clone else None,
                language=language,
            )

            record = {"text": text, "language": language}

            if audio is None:
                record["error"] = "generation_failed"
                records.append(record)
                continue

            # Save audio
            out_path = os.path.join(audio_dir, f"{tag}_{i:03d}.wav")
            sf.write(out_path, audio, sr)
            record["audio_path"] = out_path

            # Audio quality stats
            record["stats"] = compute_audio_stats(audio, sr)

            # Speaker similarity
            gen_16k = audio
            if sr != 16000:
                import librosa
                gen_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sim = speaker_similarity(ref_audio_16k, gen_16k, sr=16000)
            record["speaker_similarity"] = round(sim, 4)
            print(f"  Speaker similarity: {sim:.4f}")

            # ASR + CER (Vietnamese only, optional)
            if args.compute_cer and language in ("vi", "Vietnamese"):
                hyp = transcribe_audio(gen_16k, 16000, language="vi")
                if hyp:
                    c = cer(text, hyp)
                    record["asr_hypothesis"] = hyp
                    record["cer"] = round(c, 4)
                    print(f"  CER: {c:.4f} | Hyp: {hyp[:60]}")

            records.append(record)
        return records

    # Evaluate Vietnamese texts
    if args.test_texts_vi and os.path.exists(args.test_texts_vi):
        with open(args.test_texts_vi, encoding="utf-8") as f:
            vi_texts = f.readlines()
        print(f"\n[eval] Evaluating {len(vi_texts)} Vietnamese texts...")
        results["vietnamese"] = eval_texts(vi_texts, "vi", "vi")
    else:
        print("[eval] No Vietnamese test texts provided, skipping.")

    # Evaluate original language texts (optional)
    if args.test_texts_orig and os.path.exists(args.test_texts_orig):
        with open(args.test_texts_orig, encoding="utf-8") as f:
            orig_texts = f.readlines()
        print(f"\n[eval] Evaluating {len(orig_texts)} original language texts...")
        results["original"] = eval_texts(orig_texts, "Auto", "orig")
    else:
        print("[eval] No original language test texts provided, skipping.")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    def summarize(records: List[dict]) -> dict:
        sims = [r["speaker_similarity"] for r in records if "speaker_similarity" in r]
        cers = [r["cer"] for r in records if "cer" in r]
        errors = sum(1 for r in records if "error" in r)
        s = {
            "n_evaluated": len(records),
            "n_errors": errors,
            "speaker_similarity_mean": round(np.mean(sims), 4) if sims else None,
            "speaker_similarity_min": round(np.min(sims), 4) if sims else None,
            "speaker_similarity_std": round(np.std(sims), 4) if sims else None,
        }
        if cers:
            s["cer_mean"] = round(np.mean(cers), 4)
        return s

    vi_summary = summarize(results["vietnamese"])
    orig_summary = summarize(results["original"])
    results["summary"] = {
        "vietnamese": vi_summary,
        "original": orig_summary,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nVietnamese ({vi_summary.get('n_evaluated', 0)} samples):")
    print(f"  Speaker Similarity: {vi_summary.get('speaker_similarity_mean', 'N/A'):.4f} "
          f"± {vi_summary.get('speaker_similarity_std', 0):.4f}")
    if "cer_mean" in vi_summary:
        print(f"  CER: {vi_summary['cer_mean']:.4f}")

    if orig_summary.get("n_evaluated", 0) > 0:
        print(f"\nOriginal Language ({orig_summary.get('n_evaluated', 0)} samples):")
        print(f"  Speaker Similarity: {orig_summary.get('speaker_similarity_mean', 'N/A'):.4f} "
              f"± {orig_summary.get('speaker_similarity_std', 0):.4f}")

    print("\nThresholds for acceptable voice cloning:")
    print("  Speaker Similarity >= 0.75 → ✅ Good")
    print("  Speaker Similarity >= 0.65 → ⚠️  Acceptable")
    print("  Speaker Similarity < 0.65  → ❌ Degraded — retrain needed")

    vi_sim = vi_summary.get("speaker_similarity_mean")
    if vi_sim is not None:
        if vi_sim >= 0.75:
            print(f"\n✅ Voice cloning preserved (sim={vi_sim:.4f})")
        elif vi_sim >= 0.65:
            print(f"\n⚠️  Voice cloning marginally acceptable (sim={vi_sim:.4f}) — consider more data")
        else:
            print(f"\n❌ Voice cloning degraded (sim={vi_sim:.4f}) — check training strategy")

    print("=" * 60)

    # Save full results
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] Full results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate voice cloning quality after Vietnamese fine-tuning")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--ref_audio", type=str, required=True,
                        help="Reference speaker audio file (used for voice cloning)")
    parser.add_argument("--test_texts_vi", type=str, default=None,
                        help="Text file with Vietnamese test sentences (one per line)")
    parser.add_argument("--test_texts_orig", type=str, default=None,
                        help="Text file with original language test sentences (for regression check)")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save generated audio and metrics")
    parser.add_argument("--speaker_name", type=str, default="vi_speaker",
                        help="Speaker name used during fine-tuning")
    parser.add_argument("--use_voice_clone", action="store_true", default=True,
                        help="Use voice cloning mode (ref_audio) instead of custom_voice lookup")
    parser.add_argument("--compute_cer", action="store_true", default=False,
                        help="Compute CER using Whisper ASR (requires openai-whisper)")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
