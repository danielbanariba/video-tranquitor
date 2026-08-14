#!/usr/bin/env bash
# Instalador de video-tranquitor con detección automática de hardware.
#
# Detecta el GPU disponible y elige el perfil ideal:
#   - nvidia: NVIDIA con CUDA → ensemble (whisper.cpp CUDA + WhisperX) + diarización
#   - vulkan: Intel/AMD con Vulkan → whisper.cpp con Vulkan, sin torch
#   - cpu:    sin GPU usable → whisper.cpp con AVX2, sin torch
#
# Override manual:
#   ./install.sh --profile=nvidia|vulkan|cpu|auto
#
# Idempotente: salta los pasos ya hechos. Nunca sobrescribe .env.

set -euo pipefail

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

readonly C_BLUE="\033[1;34m"
readonly C_GREEN="\033[1;32m"
readonly C_YELLOW="\033[1;33m"
readonly C_RED="\033[1;31m"
readonly C_DIM="\033[2m"
readonly C_RESET="\033[0m"

log()  { printf "${C_BLUE}==>${C_RESET} %s\n" "$*"; }
ok()   { printf "${C_GREEN}✓${C_RESET}  %s\n" "$*"; }
warn() { printf "${C_YELLOW}⚠${C_RESET}  %s\n" "$*"; }
fail() { printf "${C_RED}✗${C_RESET}  %s\n" "$*" >&2; exit 1; }
dim()  { printf "${C_DIM}   %s${C_RESET}\n" "$*"; }

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

PROFILE="auto"

print_help() {
  cat <<EOF
Uso: $0 [--profile=auto|nvidia|vulkan|cpu]

Perfiles:
  auto    (default) Detecta hardware y elige el mejor perfil
  nvidia  Stack completo con CUDA: whisper.cpp+CUDA, WhisperX, pyannote, ensemble
  vulkan  whisper.cpp con Vulkan (Intel Iris Xe, AMD), sin stack GPU pesada
  cpu     whisper.cpp CPU+AVX2, sin GPU
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile=*) PROFILE="${1#*=}"; shift ;;
    --profile)   PROFILE="${2:-}"; shift 2 ;;
    -h|--help)   print_help; exit 0 ;;
    *)           fail "Argumento desconocido: $1 (usá --help)" ;;
  esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

readonly WHISPER_DIR="$SCRIPT_DIR/vendor/whisper.cpp"
readonly WHISPER_BIN="$WHISPER_DIR/build/bin/whisper-cli"
readonly MODEL_NAME="large-v3-turbo"
readonly MODEL_FILE="$WHISPER_DIR/models/ggml-${MODEL_NAME}.bin"

# ---------------------------------------------------------------------------
# 1. Detección de hardware
# ---------------------------------------------------------------------------

is_wsl() {
  grep -qi microsoft /proc/version 2>/dev/null
}

detect_profile() {
  # NVIDIA con CUDA: nvidia-smi responde y lista al menos un GPU.
  # (Funciona también en WSL2 — el driver vive del lado de Windows.)
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
    echo "nvidia"
    return
  fi

  # WSL2 sin NVIDIA: Vulkan va por el driver `dzn` (Vulkan→D3D12) y es
  # más lento que CPU+AVX2 en la mayoría de los casos. Defaulteamos a CPU.
  # Si el usuario quiere forzar Vulkan, puede usar --profile=vulkan.
  if is_wsl; then
    echo "cpu"
    return
  fi

  # Vulkan disponible y al menos un GPU físico (no llvmpipe software-only)
  if command -v vulkaninfo >/dev/null 2>&1; then
    local gpus
    gpus=$(vulkaninfo --summary 2>/dev/null | grep -E "^GPU[0-9]+:" || true)
    if [ -n "$gpus" ] && ! echo "$gpus" | grep -qi "llvmpipe"; then
      echo "vulkan"
      return
    fi
  fi

  # Heurística sin vulkaninfo: hay /dev/dri y no es solo software
  if [ -d /dev/dri ] && ls /dev/dri/render* >/dev/null 2>&1; then
    echo "vulkan"  # asume Vulkan disponible — instalación lo confirma
    return
  fi

  echo "cpu"
}

resolve_profile() {
  if [ "$PROFILE" = "auto" ]; then
    PROFILE=$(detect_profile)
    log "Auto-detect: perfil = ${C_GREEN}${PROFILE}${C_RESET}"
    case "$PROFILE" in
      nvidia)
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        dim "GPU: $gpu_name"
        is_wsl && dim "Detectado WSL2 — CUDA passthrough activo"
        ;;
      vulkan)
        local gpu_name
        gpu_name=$(vulkaninfo --summary 2>/dev/null | grep -A1 "deviceName" | head -1 | sed 's/.*= //' || echo "GPU Vulkan")
        dim "GPU: $gpu_name"
        ;;
      cpu)
        if is_wsl; then
          dim "WSL2 sin NVIDIA — usando CPU+AVX2 (Vulkan vía dzn rinde peor)"
          dim "Si querés forzar Vulkan igual: ./install.sh --profile=vulkan"
        else
          dim "Sin GPU detectado — modo CPU+AVX2"
        fi
        ;;
    esac
  else
    log "Perfil forzado: $PROFILE"
    is_wsl && [ "$PROFILE" = "vulkan" ] && \
      warn "WSL2 + Vulkan: el driver dzn es lento. Probá --profile=cpu si la performance no convence."
  fi

  case "$PROFILE" in
    nvidia|vulkan|cpu) ;;
    *) fail "Perfil inválido: $PROFILE. Usá: auto, nvidia, vulkan, cpu" ;;
  esac
}

# ---------------------------------------------------------------------------
# 2. Detectar distro e instalar dependencias del sistema
# ---------------------------------------------------------------------------

detect_distro() {
  if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    echo "${ID:-unknown}"
  else
    echo "unknown"
  fi
}

install_system_deps() {
  local distro
  distro=$(detect_distro)
  log "Distro detectada: $distro"

  # Paquetes base (siempre)
  local apt_pkgs="ffmpeg build-essential cmake git curl ca-certificates"
  local dnf_pkgs="ffmpeg-free gcc-c++ make cmake git curl"
  local pacman_pkgs="ffmpeg base-devel cmake git curl"

  # Paquetes específicos por perfil
  case "$PROFILE" in
    vulkan)
      apt_pkgs+=" libvulkan-dev vulkan-tools glslang-tools"
      dnf_pkgs+=" vulkan-loader-devel vulkan-headers vulkan-tools glslc"
      pacman_pkgs+=" vulkan-headers vulkan-icd-loader vulkan-tools shaderc"
      ;;
    nvidia)
      # CUDA toolkit es opcional — si falta, whisper.cpp se compila sin CUDA y
      # solo WhisperX usa la GPU. La detección de cmake hace el fallback solo.
      :
      ;;
  esac

  case "$distro" in
    ubuntu|debian|pop|linuxmint)
      log "apt-get install (sudo)..."
      sudo apt-get update -qq
      # shellcheck disable=SC2086
      sudo apt-get install -y $apt_pkgs
      [ "$PROFILE" = "vulkan" ] && (sudo apt-get install -y glslc 2>/dev/null || true)
      ;;
    fedora|rhel|centos)
      log "dnf install (sudo)..."
      # shellcheck disable=SC2086
      sudo dnf install -y $dnf_pkgs || sudo dnf install -y ${dnf_pkgs/ffmpeg-free/ffmpeg}
      ;;
    arch|manjaro|cachyos|endeavouros)
      log "pacman install (sudo)..."
      # shellcheck disable=SC2086
      sudo pacman -Sy --needed --noconfirm $pacman_pkgs
      ;;
    *)
      warn "Distro no reconocida ($distro). Instalá manualmente: ffmpeg, cmake, git, build tools."
      [ "$PROFILE" = "vulkan" ] && warn "Y también: vulkan dev headers + glslc/shaderc."
      read -rp "¿Continuar de todos modos? [y/N] " ans
      [[ "$ans" =~ ^[yY]$ ]] || fail "Abortado."
      ;;
  esac
  ok "Dependencias del sistema OK."
}

# ---------------------------------------------------------------------------
# 3. uv (gestor de Python)
# ---------------------------------------------------------------------------

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    ok "uv ya instalado ($(uv --version))"
    return
  fi
  log "Instalando uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv no quedó en PATH. Reabrí la terminal y volvé a correr."
  ok "uv instalado ($(uv --version))"
}

# ---------------------------------------------------------------------------
# 4. venv + dependencias Python (según perfil)
# ---------------------------------------------------------------------------

setup_venv() {
  local venv_was_fresh=false
  if [ ! -d "venv" ]; then
    log "Creando venv con Python 3.12..."
    uv venv --python 3.12 venv
    venv_was_fresh=true
  else
    ok "venv ya existe — no toco las dependencias instaladas"
  fi

  if [ "$venv_was_fresh" = false ]; then
    # Solo aseguramos que el paquete del proyecto esté en modo editable.
    log "Reinstalando proyecto en modo editable (no toca el resto del venv)..."
    uv pip install --python venv/bin/python -e . --quiet
    ok "Proyecto instalado."
    return
  fi

  log "Instalando dependencias del proyecto..."
  uv pip install --python venv/bin/python -e .

  case "$PROFILE" in
    nvidia)
      log "Perfil nvidia: instalando torch+CUDA, WhisperX y pyannote (~3 GB de descarga)..."
      # torch primero y desde el índice de CUDA: las ruedas con GPU no están en
      # PyPI, así que resolverlo junto al extra [gpu] traería la build de CPU.
      uv pip install --python venv/bin/python \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu128
      uv pip install --python venv/bin/python -e ".[gpu]"
      ;;
    vulkan|cpu)
      dim "Perfil $PROFILE: no se instalan torch/whisperx/pyannote (no hacen falta)"
      ;;
  esac

  ok "Dependencias Python OK."
}

# ---------------------------------------------------------------------------
# 5. Compilar whisper.cpp según perfil
# ---------------------------------------------------------------------------

build_whisper_cpp() {
  if [ -x "$WHISPER_BIN" ]; then
    ok "whisper.cpp ya compilado en $WHISPER_BIN (no rebuilding)"
    return
  fi

  if [ ! -d "$WHISPER_DIR/.git" ]; then
    log "Clonando whisper.cpp..."
    mkdir -p "$(dirname "$WHISPER_DIR")"
    git clone --depth=1 https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
  else
    ok "whisper.cpp ya clonado"
  fi

  cd "$WHISPER_DIR"
  rm -rf build

  local cmake_log=/tmp/whisper-cmake.log
  : >"$cmake_log"

  build_with() {
    local label="$1"; shift
    log "Compilando whisper.cpp: $label"
    if cmake -B build "$@" -DCMAKE_BUILD_TYPE=Release >>"$cmake_log" 2>&1 \
       && cmake --build build -j --config Release >>"$cmake_log" 2>&1; then
      ok "whisper.cpp compilado: $label"
      return 0
    fi
    rm -rf build
    return 1
  }

  case "$PROFILE" in
    nvidia)
      build_with "CUDA" -DGGML_CUDA=1 || \
      build_with "CPU+AVX2 (CUDA toolkit no disponible)" -DGGML_NATIVE=ON || \
        fail "Build falló en ambos modos. Mirá $cmake_log"
      [ "$(grep -c GGML_CUDA "$cmake_log" || true)" -gt 0 ] || \
        warn "whisper.cpp quedó sin CUDA — instalá el CUDA toolkit y borrá vendor/whisper.cpp/build para rebuildear."
      ;;
    vulkan)
      build_with "Vulkan" -DGGML_VULKAN=1 || \
      build_with "CPU+AVX2 (Vulkan no disponible)" -DGGML_NATIVE=ON || \
        fail "Build falló en ambos modos. Mirá $cmake_log"
      ;;
    cpu)
      build_with "CPU+AVX2" -DGGML_NATIVE=ON || \
        fail "Build falló. Mirá $cmake_log"
      ;;
  esac

  [ -x "$WHISPER_BIN" ] || fail "Esperaba binario en $WHISPER_BIN — no se generó."
  cd "$SCRIPT_DIR"
}

# ---------------------------------------------------------------------------
# 6. Modelo Whisper
# ---------------------------------------------------------------------------

download_model() {
  if [ -f "$MODEL_FILE" ]; then
    ok "Modelo $MODEL_NAME ya descargado ($(du -h "$MODEL_FILE" | cut -f1))"
    return
  fi
  log "Descargando modelo $MODEL_NAME (~1.5 GB)..."
  cd "$WHISPER_DIR"
  bash models/download-ggml-model.sh "$MODEL_NAME"
  cd "$SCRIPT_DIR"
  [ -f "$MODEL_FILE" ] || fail "Modelo no se descargó correctamente."
  ok "Modelo descargado."
}

# ---------------------------------------------------------------------------
# 7. Generar .env según perfil
# ---------------------------------------------------------------------------

write_env() {
  if [ -f .env ]; then
    warn ".env ya existe — no lo sobrescribo (revisalo a mano si hace falta)"
    return
  fi

  log "Generando .env para perfil: $PROFILE"

  local has_codex="false"
  command -v codex >/dev/null 2>&1 && has_codex="true"

  case "$PROFILE" in
    nvidia)
      cat > .env <<EOF
# ============================================================================
# video-tranquitor — perfil NVIDIA (CUDA)
# Generado por install.sh el $(date '+%Y-%m-%d %H:%M:%S')
# ============================================================================

# Ensemble: corre whisper.cpp turbo + WhisperX en paralelo y arbitra con Codex
TRANSCRIBER=ensemble
WHISPER_CPP_PATH=$WHISPER_BIN
WHISPER_MODEL_PATH=$MODEL_FILE
WHISPERX_MODEL=large-v3

# Diarización de hablantes (requiere HF_TOKEN — conseguilo en
# https://huggingface.co/settings/tokens y aceptá el modelo
# pyannote/speaker-diarization-3.1)
HF_TOKEN=
ENABLE_DIARIZATION=false

# Análisis con IA: requiere Codex CLI instalada y autenticada
ENABLE_ANALYSIS=$has_codex

# Salida
ENABLE_TOON=true
ENABLE_OBSIDIAN=false
# OBSIDIAN_VAULT_PATH=/ruta/a/tu/vault

# Directorios
WATCH_DIR=$SCRIPT_DIR/Audios
OUTPUT_DIR=$SCRIPT_DIR/output

# Idioma y sample rate
LANGUAGE=es
TARGET_SAMPLE_RATE=16000

# Solo necesaria si cambiás TRANSCRIBER=openai
# OPENAI_API_KEY=
EOF
      [ "$has_codex" = "false" ] && \
        warn "Codex CLI no detectado — análisis quedó OFF. Instalalo y poné ENABLE_ANALYSIS=true."
      ;;
    vulkan|cpu)
      cat > .env <<EOF
# ============================================================================
# video-tranquitor — perfil $PROFILE (100% local, sin nube)
# Generado por install.sh el $(date '+%Y-%m-%d %H:%M:%S')
# ============================================================================

TRANSCRIBER=local
WHISPER_CPP_PATH=$WHISPER_BIN
WHISPER_MODEL_PATH=$MODEL_FILE

ENABLE_DIARIZATION=false
ENABLE_ANALYSIS=false
ENABLE_OBSIDIAN=false
ENABLE_TOON=true

WATCH_DIR=$SCRIPT_DIR/Audios
OUTPUT_DIR=$SCRIPT_DIR/output
LANGUAGE=es
TARGET_SAMPLE_RATE=16000
EOF
      ;;
  esac
  ok ".env generado."
}

# ---------------------------------------------------------------------------
# 8. Directorios de trabajo
# ---------------------------------------------------------------------------

create_dirs() {
  mkdir -p Audios output
  ok "Directorios Audios/ y output/ listos."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
  log "Iniciando instalación de video-tranquitor"
  echo

  resolve_profile
  install_system_deps
  install_uv
  setup_venv
  build_whisper_cpp
  download_model
  write_env
  create_dirs

  echo
  ok "Instalación completa (perfil: $PROFILE)"
  echo
  echo "Próximos pasos:"
  echo "  1. source venv/bin/activate"
  echo "  2. make start                       # daemon (escucha Audios/)"
  echo "     o: make process FILE=video.mp4   # single-shot"
  echo
  case "$PROFILE" in
    nvidia)
      echo "Notas del perfil nvidia:"
      echo "  - Para diarización: editá HF_TOKEN en .env y poné ENABLE_DIARIZATION=true"
      [ "$(command -v codex >/dev/null 2>&1 && echo y || echo n)" = "n" ] && \
        echo "  - Para análisis con IA: instalá el Codex CLI y poné ENABLE_ANALYSIS=true"
      ;;
    vulkan)
      echo "Nota: si la velocidad no te convence, probá un modelo más chico:"
      echo "  cd vendor/whisper.cpp && bash models/download-ggml-model.sh small"
      echo "  y actualizá WHISPER_MODEL_PATH en .env"
      ;;
  esac
  echo
  echo "Salida en: $SCRIPT_DIR/output/"
}

main "$@"
