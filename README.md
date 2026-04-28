# video-tranquitor

Pipeline de transcripción de audio/video al español. Tira un archivo en una carpeta y te devuelve la transcripción con timestamps. Funciona **100% local** (sin nube, sin API keys) si tu hardware lo permite, o aprovecha NVIDIA + nube para máxima precisión cuando está disponible.

## Qué hace

1. **Detecta** archivos nuevos en `Audios/` (modo daemon) o procesa uno solo (modo CLI).
2. **Preprocesa** el audio con `ffmpeg`: extrae el audio del video, lo pasa a 16 kHz mono, le aplica filtros de limpieza (highpass, lowpass, denoise, loudnorm).
3. **Transcribe** con el motor que mejor se adapte a tu hardware (ver "perfiles" abajo).
4. **Diariza** hablantes (opcional, requiere NVIDIA).
5. **Analiza** con IA (opcional): resumen, requerimientos, accionables, decisiones.
6. **Escribe** la salida como TOON tabular y/o nota Markdown para Obsidian.

Extensiones soportadas: `.mp4`, `.mkv`, `.avi`, `.mov`, `.ogg`, `.mp3`, `.wav`, `.m4a`, `.flac`.

## Instalación

**Un solo comando:**

```bash
git clone <url-del-repo> video-tranquitor
cd video-tranquitor
./install.sh
```

El script detecta tu hardware automáticamente y configura el perfil ideal. Es **idempotente**: lo podés correr varias veces, salta los pasos que ya están hechos. **Nunca sobrescribe `.env`** — si ya existe, lo deja intacto.

### Perfiles (auto-detectados)

| Perfil | Cuándo se elige | Qué instala | Qué configura |
|--------|-----------------|-------------|---------------|
| `nvidia` | `nvidia-smi` detecta una GPU NVIDIA | torch+CUDA, WhisperX, pyannote.audio + whisper.cpp con CUDA | `TRANSCRIBER=ensemble` (whisper.cpp turbo + WhisperX large-v3, arbitrados con Codex) |
| `vulkan` | GPU Intel Iris Xe / AMD con Vulkan disponible | whisper.cpp con backend Vulkan, **sin** torch | `TRANSCRIBER=local` (100% local, sin nube) |
| `cpu` | Sin GPU usable | whisper.cpp CPU+AVX2, **sin** torch | `TRANSCRIBER=local` (100% local, sin nube) |

**Forzar un perfil específico:**

```bash
./install.sh --profile=nvidia   # ignorar detección, instalar stack NVIDIA
./install.sh --profile=vulkan   # forzar Vulkan
./install.sh --profile=cpu      # CPU puro, lo más liviano
./install.sh --profile=auto     # default — detecta solo
```

### Lo que hace en cualquier perfil

- Instala dependencias del sistema (`ffmpeg`, `cmake`, build tools, y headers de GPU según perfil) con el package manager de tu distro.
- Instala [`uv`](https://github.com/astral-sh/uv) si no lo tenés.
- Crea un venv con Python 3.12.
- Clona y compila whisper.cpp con el backend correcto.
- Descarga el modelo `ggml-large-v3-turbo` (~1.5 GB).
- Genera un `.env` adecuado al perfil.

## Requisitos

- **Linux** — probado en Ubuntu/Debian/Pop, Fedora/RHEL, Arch/CachyOS/Manjaro.
- **CPU x86_64 con AVX2** (cualquier Intel/AMD de los últimos 10 años cumple).
- **8 GB RAM** mínimo (16 GB si vas perfil NVIDIA con WhisperX).
- **~3 GB libres** (5 GB para perfil NVIDIA por torch+CUDA wheels).
- **GPU opcional** — el proyecto funciona sin GPU, solo más lento.

No requiere conexión a internet **después** de la instalación (excepto si activás análisis con Codex).

## Uso

### Modo daemon (recomendado)

```bash
source venv/bin/activate
make start
```

Queda escuchando la carpeta `Audios/`. Tirale archivos ahí y los va procesando uno por uno. La salida queda en `output/{nombre}_transcription.toon`. `Ctrl+C` para parar.

### Modo single-shot

```bash
source venv/bin/activate
make process FILE=/ruta/al/archivo.mp4
```

### Ejemplo de salida (`output/reunion_transcription.toon`)

```
transcripciones[3]{inicio,fin,texto}:
  "00:00:00","00:02:00","Buenos días equipo, arrancamos con el daily..."
  "00:02:00","00:04:00","Sobre el ticket de autenticación, lo terminé ayer..."
  "00:04:00","00:05:32","Cualquier cosa me avisan. Cierro reunión."
```

## Configuración

Toda la configuración vive en `.env`. El instalador lo genera con valores razonables. Los más interesantes:

| Variable | Default (modo local) | Para qué sirve |
|----------|----------------------|----------------|
| `TRANSCRIBER` | `local` | `local` (whisper.cpp) o `openai` (API en la nube — requiere `OPENAI_API_KEY`). |
| `WHISPER_CPP_PATH` | autogenerado | Ruta al binario `whisper-cli`. |
| `WHISPER_MODEL_PATH` | autogenerado | Ruta al `.bin` del modelo. |
| `WATCH_DIR` | `./Audios` | Carpeta que monitorea el daemon. |
| `OUTPUT_DIR` | `./output` | Donde se escriben las transcripciones. |
| `LANGUAGE` | `es` | Idioma del audio. |
| `ENABLE_TOON` | `true` | Escribir el archivo `.toon`. |
| `ENABLE_DIARIZATION` | `false` | Identificar quién habla (ver abajo). |
| `ENABLE_ANALYSIS` | `false` | Análisis con IA — extrae requerimientos/accionables (ver abajo). |
| `ENABLE_OBSIDIAN` | `false` | Generar nota Markdown en un vault de Obsidian. |

## Cambiar el modelo

`large-v3-turbo` es el default — buen balance. Si querés algo más liviano (más rápido pero menos preciso) o más pesado (más lento pero más preciso):

```bash
cd vendor/whisper.cpp
bash models/download-ggml-model.sh tiny      # ~75 MB,  rapidísimo, calidad baja
bash models/download-ggml-model.sh base      # ~140 MB, rápido,      calidad media
bash models/download-ggml-model.sh small     # ~470 MB, decente,     calidad media-alta
bash models/download-ggml-model.sh medium    # ~1.5 GB, lento,       calidad alta
bash models/download-ggml-model.sh large-v3  # ~3 GB,   muy lento,   máxima calidad
```

Después editá `WHISPER_MODEL_PATH` en `.env` apuntando al nuevo `.bin`.

## Features opcionales

### Diarización (identificar hablantes)

Identifica quién dice qué (`SPEAKER_00`, `SPEAKER_01`, etc.). **Solo recomendado con perfil `nvidia`** — en CPU es inutilizablemente lento.

1. Si te instalaste con `--profile=nvidia`, ya tenés torch+pyannote listos.
2. Conseguí un token en [Hugging Face](https://huggingface.co/settings/tokens) y aceptá los términos del modelo [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1).
3. En `.env`: `HF_TOKEN=hf_...` y `ENABLE_DIARIZATION=true`.

### Análisis con IA

Genera resumen + lista de requerimientos + accionables + decisiones a partir de la transcripción. Requiere el [Codex CLI](https://github.com/openai/codex) instalado y autenticado:

```bash
# En .env:
ENABLE_ANALYSIS=true
```

### Notas en Obsidian

Genera un `.md` por reunión, con frontmatter YAML, en el vault que vos digas:

```bash
# En .env:
ENABLE_OBSIDIAN=true
OBSIDIAN_VAULT_PATH=/ruta/a/tu/vault/Reuniones
```

## Comandos útiles

```bash
make start                # daemon (escucha WATCH_DIR)
make process FILE=...     # single-shot
make test                 # pytest
make lint                 # ruff check
make format               # ruff format
```

## Troubleshooting

**`E_WHISPER_NOT_FOUND`**: el binario `whisper-cli` no existe en el path configurado. Reejecutá `./install.sh` o verificá `WHISPER_CPP_PATH` en `.env`.

**`ffmpeg: command not found`**: el instalador no detectó tu distro. Instalalo a mano (`sudo apt install ffmpeg` o equivalente).

**El build de whisper.cpp con Vulkan falla**: el script automáticamente cae a CPU-only. Funciona igual, solo más lento. Mirá `/tmp/whisper-cmake.log` si querés saber por qué falló Vulkan.

**Transcripción muy lenta**: probá un modelo más chico (`small` o `base`). En CPU pura, `large-v3-turbo` corre alrededor de 0.5× tiempo real (10 min de audio → ~20 min de procesamiento). Con Vulkan en Iris Xe, alrededor de 1.5-2× tiempo real.

**El daemon no detecta archivos**: el watcher ignora archivos que empiezan con `.` o `temp_`. Si copiás un archivo grande, esperá a que termine de copiarse — algunos sistemas hacen el rename atómico al final.

## Licencia y créditos

Usa whisper.cpp (MIT) y los modelos de OpenAI Whisper (MIT). El proyecto en sí no tiene licencia declarada.
