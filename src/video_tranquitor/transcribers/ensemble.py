"""Orquestador ensemble: corre whisper.cpp turbo + WhisperX y arbitra con Codex."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from video_tranquitor.llm_client import call_llm_with_schema
from video_tranquitor.transcribers.whispercpp import (
    transcribe_local,
    whisper_result_to_transcriptions,
)
from video_tranquitor.transcribers.whisperx import (
    whisperx_result_to_transcriptions,
)
from video_tranquitor.types import EnsembleResult, PipelineConfig, Transcription, WhisperResult

logger = logging.getLogger(__name__)

def _run_whisperx_in_subprocess(
    audio_path: str, config_json: str, model_size: str
) -> dict:
    """Wrapper picklable para correr WhisperX en un proceso separado.

    Recibe el config serializado como JSON (evita problemas de pickling de pydantic
    con tipos complejos en algunos entornos), reconstruye el modelo adentro del
    worker, y retorna el WhisperResult serializado como dict.
    """
    from video_tranquitor.transcribers.whisperx import transcribe_whisperx as _tx
    from video_tranquitor.types import PipelineConfig as _Cfg

    cfg = _Cfg.model_validate_json(config_json)
    result = _tx(audio_path, cfg, model_size)
    return result.model_dump()


ENSEMBLE_SCHEMA = {
    "type": "object",
    "properties": {
        "transcripciones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "inicio": {"type": "string"},
                    "fin": {"type": "string"},
                    "texto": {"type": "string"},
                },
                "required": ["inicio", "fin", "texto"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["transcripciones"],
    "additionalProperties": False,
}


def _build_arbitration_prompt(toon_turbo: str, toon_whisperx: str) -> str:
    return (
        "Sos un experto en transcripción de audio en español. Te paso DOS transcripciones "
        "del mismo audio en formato TOON (tabular), una de whisper.cpp turbo y otra de "
        "WhisperX con large-v3.\n\n"
        "Tu tarea: producir UNA transcripción consensuada, chunk por chunk, eligiendo la "
        "mejor versión basándote en el CONTEXTO global de la reunión.\n\n"
        "Reglas:\n"
        "- Cada chunk tiene los mismos timestamps (inicio, fin) en ambas transcripciones — "
        "conservalos EXACTOS\n"
        "- Cuando difieran en un nombre propio, elegí el que sea consistente con el resto "
        'de la reunión (ej: si "Edwin" aparece 10 veces y "Edo" solo 1, probablemente era '
        '"Edwin" en todos los casos)\n'
        "- Cuando difieran en términos técnicos, elegí el que tenga sentido en el contexto\n"
        "- Cuando una versión tenga duplicaciones (mismo texto en 2 chunks seguidos), "
        "preferí la otra\n"
        "- Si no podés decidir con certeza, usá la versión de WhisperX (es más precisa en "
        "general)\n"
        "- NO inventes texto nuevo: solo elegí entre las dos versiones existentes, o "
        "combinalas palabra por palabra\n"
        "- NO cambies los timestamps\n\n"
        "Devolvé el resultado como JSON con la estructura: "
        '{ "transcripciones": [{inicio, fin, texto}] }\n\n'
        f"=== TURBO (whisper.cpp) ===\n{toon_turbo}\n\n"
        f"=== WHISPERX (large-v3) ===\n{toon_whisperx}"
    )


def _validate_arbitration_result(parsed: object) -> dict:
    p = parsed
    if not isinstance(p, dict) or not isinstance(p.get("transcripciones"), list):
        raise ValueError("Estructura JSON inválida: falta 'transcripciones'")
    return p


def _validate_and_snap_timestamps(
    arbitrated: list[Transcription],
    turbo: list[Transcription],
    whisperx: list[Transcription],
) -> list[Transcription]:
    """Valida que los timestamps arbitrados coincidan con turbo o whisperx.

    Si un chunk no coincide con ninguno, lo ajusta al chunk de whisperx del
    mismo índice (si existe).
    """
    turbo_set = {(t.inicio, t.fin) for t in turbo}
    whisperx_set = {(w.inicio, w.fin) for w in whisperx}

    result: list[Transcription] = []
    for idx, chunk in enumerate(arbitrated):
        key = (chunk.inicio, chunk.fin)
        if key in turbo_set or key in whisperx_set:
            result.append(chunk)
            continue

        # Ajustar al chunk de whisperx del mismo índice
        if idx < len(whisperx):
            wx_chunk = whisperx[idx]
            result.append(
                Transcription(
                    inicio=wx_chunk.inicio,
                    fin=wx_chunk.fin,
                    texto=chunk.texto,
                )
            )
        else:
            result.append(chunk)

    return result


async def transcribe_ensemble(
    audio_path: str,
    config: PipelineConfig,
) -> EnsembleResult:
    """Ejecuta whisper.cpp turbo + WhisperX en paralelo y arbitra con Codex.

    Los cuatro casos posibles:
    - Ambos fallan: lanza RuntimeError.
    - Solo turbo: devuelve EnsembleResult sin arbitración.
    - Solo WhisperX: devuelve EnsembleResult sin arbitración.
    - Ambos OK: codifica en TOON, llama a Codex para arbitrar y valida timestamps.

    Args:
        audio_path: Ruta al archivo de audio WAV.
        config:     Configuración del pipeline.

    Returns:
        EnsembleResult con los chunks arbitrados (o fallback al mejor disponible).
    """
    print(
        "Ensemble: corriendo whisper.cpp turbo + WhisperX en paralelo "
        "(procesos separados, CUDA context aislado)..."
    )

    loop = asyncio.get_running_loop()
    # spawn context → cada proceso arranca fresh, sin heredar CUDA state del padre
    spawn_ctx = multiprocessing.get_context("spawn")

    # turbo corre en thread (thin subprocess wrapper, sin carga de Python GPU)
    turbo_task = asyncio.to_thread(transcribe_local, audio_path, config)

    # whisperx corre en su propio proceso (GPU-pesado en Python)
    with ProcessPoolExecutor(max_workers=1, mp_context=spawn_ctx) as pool:
        whisperx_future = loop.run_in_executor(
            pool,
            _run_whisperx_in_subprocess,
            audio_path,
            config.model_dump_json(),
            config.whisperx_model,
        )

        turbo_settled_raw, whisperx_raw = await asyncio.gather(
            turbo_task, whisperx_future, return_exceptions=True
        )

    # Reconstruir WhisperResult desde el dict serializado
    turbo_settled = turbo_settled_raw
    if isinstance(whisperx_raw, dict):
        whisperx_settled: WhisperResult | BaseException = WhisperResult.model_validate(whisperx_raw)
    else:
        whisperx_settled = whisperx_raw  # excepción

    turbo_ok = not isinstance(turbo_settled, BaseException)
    whisperx_ok = not isinstance(whisperx_settled, BaseException)

    # Caso 1: ambos fallaron
    if not turbo_ok and not whisperx_ok:
        raise RuntimeError(
            "Ensemble: ambos transcriptores fallaron.\n"
            f"  whisper.cpp: {turbo_settled}\n"
            f"  WhisperX: {whisperx_settled}"
        )

    # Caso 2: solo turbo OK
    if turbo_ok and not whisperx_ok:
        logger.warning(
            "Ensemble: WhisperX falló (%s). Usando solo whisper.cpp turbo.", whisperx_settled
        )
        print("  Ensemble: WhisperX falló. Usando solo whisper.cpp turbo.")
        turbo_chunks = whisper_result_to_transcriptions(turbo_settled)  # type: ignore[arg-type]
        return EnsembleResult(
            whisper_result=turbo_settled,  # type: ignore[arg-type]
            arbitrated=turbo_chunks,
            turbo=turbo_chunks,
            whisperx=[],
            arbitration_used=False,
        )

    # Caso 3: solo WhisperX OK
    if not turbo_ok and whisperx_ok:
        logger.warning(
            "Ensemble: whisper.cpp falló (%s). Usando solo WhisperX.", turbo_settled
        )
        print("  Ensemble: whisper.cpp falló. Usando solo WhisperX.")
        whisperx_chunks = whisperx_result_to_transcriptions(whisperx_settled, 120)  # type: ignore[arg-type]
        return EnsembleResult(
            whisper_result=whisperx_settled,  # type: ignore[arg-type]
            arbitrated=whisperx_chunks,
            turbo=[],
            whisperx=whisperx_chunks,
            arbitration_used=False,
        )

    # Caso 4: ambos OK — arbitración con el proveedor configurado
    # (narrowing garantizado por los checks anteriores)
    from video_tranquitor.writers.toon_writer import encode  # noqa: PLC0415 — importación tardía

    turbo_chunks = whisper_result_to_transcriptions(turbo_settled)  # type: ignore[arg-type]
    whisperx_chunks = whisperx_result_to_transcriptions(whisperx_settled, 120)  # type: ignore[arg-type]

    toon_turbo = encode({"turbo": [c.model_dump() for c in turbo_chunks]})
    toon_whisperx = encode({"whisperx": [c.model_dump() for c in whisperx_chunks]})

    arbitration_prompt = _build_arbitration_prompt(toon_turbo, toon_whisperx)

    arbitration_response = await call_llm_with_schema(
        prompt=arbitration_prompt,
        schema=ENSEMBLE_SCHEMA,
        validate=_validate_arbitration_result,
        provider=config.analysis_provider,
        model=config.analysis_model,
        effort=config.analysis_effort,
        max_retries=3,
        timeout_sec=20 * 60,
        error_code="E_ARBITRATION_FAILED",
    )

    if arbitration_response is None:
        print("  Ensemble: arbitración falló. Usando WhisperX como fallback.")
        return EnsembleResult(
            whisper_result=whisperx_settled,  # type: ignore[arg-type]
            arbitrated=whisperx_chunks,
            turbo=turbo_chunks,
            whisperx=whisperx_chunks,
            arbitration_used=False,
        )

    raw_items: list[dict] = arbitration_response["transcripciones"]
    arbitrated_transcriptions = [
        Transcription(inicio=item["inicio"], fin=item["fin"], texto=item["texto"])
        for item in raw_items
    ]

    validated = _validate_and_snap_timestamps(
        arbitrated_transcriptions, turbo_chunks, whisperx_chunks
    )

    return EnsembleResult(
        whisper_result=whisperx_settled,  # type: ignore[arg-type]
        arbitrated=validated,
        turbo=turbo_chunks,
        whisperx=whisperx_chunks,
        arbitration_used=True,
    )
