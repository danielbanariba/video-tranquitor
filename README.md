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
| `nvidia` | `nvidia-smi` detecta una GPU NVIDIA | torch+CUDA, WhisperX, pyannote.audio + whisper.cpp con CUDA | `TRANSCRIBER=whisperx` (WhisperX large-v3, con diarización disponible) |
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

## Instalación en Windows (vía WSL2)

El `install.sh` es bash y depende de herramientas Linux. **No corre nativamente en Windows**, pero corre perfectamente en WSL2 (Linux dentro de Windows). El proceso es:

### Paso 1 — Instalar WSL2 + Ubuntu

Abrí **PowerShell como Administrador** (click derecho → "Ejecutar como administrador") y corré:

```powershell
wsl --install -d Ubuntu-24.04
```

Eso instala WSL2 y Ubuntu 24.04. **Reiniciá la PC** cuando termine. Al volver a iniciar, se va a abrir una terminal de Ubuntu pidiéndote crear un usuario y contraseña — ponele lo que quieras (esto es independiente de tu cuenta de Windows).

Si ya tenés WSL pero no Ubuntu 24.04, listá distros con `wsl -l -o` e instalá la que prefieras (`Ubuntu-22.04` también funciona).

### Paso 2 — Instalar el proyecto dentro de Ubuntu

Una vez en la terminal de Ubuntu (la abrís desde el menú Inicio escribiendo "Ubuntu"), corré los comandos normales:

```bash
sudo apt update && sudo apt upgrade -y
git clone <url-del-repo> video-tranquitor
cd video-tranquitor
./install.sh
```

`install.sh` detecta que está en WSL2 y elige el perfil correcto automáticamente.

### Paso 3 — Usar el proyecto

Igual que en Linux nativo:

```bash
source venv/bin/activate
make start
```

### Cómo pasar archivos entre Windows y WSL

Los archivos en `Audios/` viven dentro de Linux pero son accesibles desde Windows. Tres formas:

**A. Desde el Explorador de Windows** — escribí en la barra de direcciones:
```
\\wsl$\Ubuntu-24.04\home\TU_USUARIO\video-tranquitor\Audios
```
Podés arrastrar archivos ahí como cualquier carpeta normal.

**B. Desde la terminal de Ubuntu** — abrí la carpeta actual en el Explorador:
```bash
explorer.exe .
```

**C. Desde PowerShell** — copiá un archivo de Windows a WSL:
```powershell
wsl cp "C:\Users\TuUsuario\Videos\reunion.mp4" "/home/TU_USUARIO/video-tranquitor/Audios/"
```

### Notas de performance en WSL2

- **NVIDIA + CUDA** funciona perfecto en WSL2 (Microsoft + NVIDIA lo soportan oficialmente). El driver vive del lado de Windows y se expone automáticamente. `nvidia-smi` funciona dentro de WSL.
- **Intel Iris Xe / AMD integrado**: Vulkan en WSL2 va por el driver `dzn` (Vulkan→D3D12), que es **más lento que CPU+AVX2** en muchos casos. Si tu amigo tiene Iris Xe, recomiendo forzar el perfil CPU:
  ```bash
  ./install.sh --profile=cpu
  ```
- Para el i7-1360P + CPU+AVX2 con `large-v3-turbo`: ~30 min de audio se procesan en ~45-60 min. Si es muy lento, bajale al modelo (`small` procesa en ~5-10 min con calidad aceptable).

## Requisitos

- **Linux nativo** o **Windows 11 con WSL2** (Windows 10 22H2 también sirve).
- **CPU x86_64 con AVX2** (cualquier Intel/AMD de los últimos 10 años cumple).
- **8 GB RAM** mínimo (16 GB si vas perfil NVIDIA con WhisperX).
- **~3 GB libres** (5 GB para perfil NVIDIA por torch+CUDA wheels). En Windows, sumá ~10 GB para WSL+Ubuntu.
- **GPU opcional** — el proyecto funciona sin GPU, solo más lento.

Distros Linux probadas: Ubuntu/Debian/Pop, Fedora/RHEL, Arch/CachyOS/Manjaro.

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

Hay una plantilla completa y comentada en [`.env.example`](./.env.example). Las variables principales:

| Variable | Default | Para qué sirve |
|----------|---------|----------------|
| `TRANSCRIBER` | `local` | `local` (whisper.cpp), `whisperx` (recomendada), `openai` (API paga) o `ensemble`. |
| `WHISPER_CPP_PATH` | autogenerado | Ruta al binario `whisper-cli`. Solo para `local` y `ensemble`. |
| `WHISPER_MODEL_PATH` | autogenerado | Ruta al `.bin` del modelo. Solo para `local` y `ensemble`. |
| `WHISPERX_MODEL` | `large-v3` | Modelo de WhisperX. |
| `WATCH_DIR` | `./Audios` | Carpeta que monitorea el daemon. |
| `OUTPUT_DIR` | `./output` | Donde se escriben las transcripciones. |
| `LANGUAGE` | `es` | Idioma del audio. |
| `ENABLE_TOON` | `true` | Escribir el archivo `.toon`. |
| `ENABLE_DIARIZATION` | `false` | Identificar quién habla (ver abajo). Requiere `HF_TOKEN`. |
| `ENABLE_ANALYSIS` | `true` | Análisis con IA — extrae requerimientos/accionables (ver abajo). |
| `ENABLE_OBSIDIAN` | `true` | Generar nota Markdown en un vault de Obsidian. |
| `ANALYSIS_PROVIDER` | `codex` | `codex` (CLI de Codex) o `claude` (CLI de Claude Code). |
| `ANALYSIS_PASSES` | `1` | Pasadas del análisis que después se unen. Ver abajo. |

### Por qué conviene `TRANSCRIBER=whisperx`

El modo `ensemble` corre whisper.cpp y WhisperX en paralelo y arbitra con un LLM. Suena mejor y no lo
es. Medido sobre una reunión completa: de 8 chunks arbitrados, 8 se parecían a WhisperX y ninguno a
whisper.cpp, o sea que el árbitro reproduce WhisperX con 195 s de trabajo que WhisperX solo hace en
22 s. Y donde sí se desvía, empeora: `Mongo-dbcrm-hn` salió como `Mongo guión bajo DB CRM guión bajo
HN`, inservible para pegar en una consulta.

### Por qué subir `ANALYSIS_PASSES`

El modelo no es determinista. Con la misma transcripción de entrada, una corrida captura el nombre
exacto de una base de datos y la siguiente lo omite. Con 3 pasadas que después se unen, sobre la misma
reunión: requerimientos 6 → 10, decisiones 6 → 8, y el día del que trataba toda la reunión pasó de 0
a 10 menciones. Cuesta unos 130 s.

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
2. Aceptá las condiciones del modelo [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) y generá un token de tipo **Read** en [Hugging Face](https://huggingface.co/settings/tokens). El orden importa: sin aceptar las condiciones, el token no sirve y la descarga falla con 401.
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
