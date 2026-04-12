#!/usr/bin/env python3
"""
WhisperX transcription sidecar.

Usage:
    python transcribe_whisperx.py <audio_path> <model_size> <language> [output_json]

Outputs JSON to stdout (or to output_json path if provided) with format:
{
  "segments": [
    {
      "text": "...",
      "start": 0.0,
      "end": 5.0,
      "words": [
        {"word": "hola", "start": 0.1, "end": 0.3}
      ]
    }
  ],
  "language": "es"
}
"""

import sys
import json
import os
import gc

def main():
    if len(sys.argv) < 4:
        print("Usage: transcribe_whisperx.py <audio_path> <model_size> <language> [output_json]", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]
    model_size = sys.argv[2]
    language = sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) > 4 else None

    if not os.path.exists(audio_path):
        print(f"E_AUDIO_NOT_FOUND: {audio_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        import whisperx
    except ImportError as e:
        print(f"E_WHISPERX_NOT_INSTALLED: {e}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    batch_size = 16 if device == "cuda" else 4

    print(f"[whisperx] Device: {device}, compute: {compute_type}, model: {model_size}", file=sys.stderr)

    # Stage 1: Load model + transcribe (with internal VAD)
    print("[whisperx] Cargando modelo...", file=sys.stderr)
    model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)

    print("[whisperx] Cargando audio...", file=sys.stderr)
    audio = whisperx.load_audio(audio_path)

    print("[whisperx] Transcribiendo (con VAD integrado)...", file=sys.stderr)
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    # Free memory
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Stage 2: Forced alignment for word-level timestamps
    print("[whisperx] Alineando palabras (wav2vec2)...", file=sys.stderr)
    try:
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False
        )
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[whisperx] Alineación falló, usando timestamps de Whisper: {e}", file=sys.stderr)

    # Normalize to our expected format
    output = {
        "segments": [],
        "language": language,
    }

    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })
        output["segments"].append({
            "text": seg.get("text", "").strip(),
            "start": float(seg.get("start", 0)),
            "end": float(seg.get("end", 0)),
            "words": words,
        })

    json_out = json.dumps(output, ensure_ascii=False)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_out)
        print(f"[whisperx] Guardado en: {output_path}", file=sys.stderr)
    else:
        print(json_out)

    print("[whisperx] Completado.", file=sys.stderr)


if __name__ == "__main__":
    main()
