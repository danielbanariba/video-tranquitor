#!/usr/bin/env python3
"""
Speaker diarization using pyannote-audio 4.x.
Outputs a JSON array to stdout: [{"speaker": "SPEAKER_00", "start": 1.23, "end": 4.56}, ...]
Sorted by start time.
"""

import json
import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: diarize.py <audio_path>", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("E_HF_TOKEN_MISSING", file=sys.stderr)
        sys.exit(1)

    # Import heavy deps after arg validation to fail fast on missing token
    import torch  # type: ignore
    from pyannote.audio import Pipeline  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(device)

    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            }
        )

    segments.sort(key=lambda s: s["start"])

    print(json.dumps(segments, ensure_ascii=False))


if __name__ == "__main__":
    main()
