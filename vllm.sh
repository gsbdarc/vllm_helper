# vllm.sh
#!/usr/bin/env bash
# Unified vLLM helper for Apptainer/Singularity on HPC (Yens + Sherlock)
# - Starts an OpenAI-compatible vLLM server in a container
# - Pins HF + compile caches to $SCRATCH_BASE
# - Gives container a writable HOME via --home without overriding HOME env
#
# Usage:
#   export SCRATCH_BASE=/path/to/scratch
#   export VLLM_MODEL=Qwen/Qwen3-8B-Base
#   # Optional:
#   #   VLLM_SIF=/path/to/vllm-openai.sif
#   #   VLLM_TP=2
#   #   VLLM_MAX_LEN=16384
#   #   VLLM_QUANT=fp8
#   #   VLLM_ENABLE_LORA=1; VLLM_LORAS="my-lora=/models/my-lora"
#   #   VLLM_ENFORCE_EAGER=1
#   #   HF_TOKEN=xxxxxxxx
#   #   VLLM_DISABLE_FLASHINFER_JIT=1
#   #   APPTAINER_BIN=/usr/bin/apptainer
#
#   source ./vllm.sh
#   vllm serve &
#   vllm chat "Hello"
#   vllm stop

vllm() {
  local LOWERPORT=32768 UPPERPORT=60999
  local S PORT_FILE HOST_FILE PORT HOST SIF MODEL BIND ENV_HOST APPT

  # Require SCRATCH_BASE
  if [ -z "${SCRATCH_BASE:-}" ]; then
    echo "ERROR: please export SCRATCH_BASE" >&2
    return 1
  fi

  # Require model
  if [ -z "${VLLM_MODEL:-}" ]; then
    echo "ERROR: please export VLLM_MODEL (e.g., export VLLM_MODEL=Qwen/Qwen3-8B-Base)" >&2
    return 1
  fi

  # Find apptainer/singularity
  APPT="${APPTAINER_BIN:-}"
  if [ -z "$APPT" ]; then
    if command -v apptainer >/dev/null 2>&1; then
      APPT=$(command -v apptainer)
    elif command -v singularity >/dev/null 2>&1; then
      APPT=$(command -v singularity)
    else
      echo "ERROR: apptainer/singularity not found in PATH" >&2
      return 1
    fi
  fi

  # Scratch/caches (no global env; container sees these via binds)
  S="${SCRATCH_BASE}/vllm"
  mkdir -p "${S}"/{models,hf-cache,tmp,cache,home}
  mkdir -p "${S}/cache"/{flashinfer,torchinductor,triton,nv}
  PORT_FILE="${S}/port.txt"
  HOST_FILE="${S}/host.txt"

  # Container SIF (required or default to local)
  SIF="${VLLM_SIF:-$PWD/vllm-openai.sif}"
  MODEL="${VLLM_MODEL}"

  # Helper: pick an unused TCP port
  find_available_port() {
    local p
    while :; do
      p=$(shuf -i "${LOWERPORT}-${UPPERPORT}" -n1)
      if ! ss -tuln | grep -q ":${p} "; then
        echo "$p"; return
      fi
    done
  }

  if [ "$1" = "serve" ] || [ ! -f "$PORT_FILE" ] || [ ! -f "$HOST_FILE" ]; then
    # Start server
    PORT=$(find_available_port)
    echo "$PORT" > "$PORT_FILE"
    hostname -s > "$HOST_FILE"

    BIND="0.0.0.0:${PORT}"
    ENV_HOST="http://$(<"$HOST_FILE"):${PORT}"

    echo "Starting vLLM server binding to ${BIND}"
    echo "Advertising server to clients at ${ENV_HOST}"
    shift

    unset ROCR_VISIBLE_DEVICES

    # Optional FlashInfer JIT toggle
    local FLASHINFER_ARGS=()
    if [ "${VLLM_DISABLE_FLASHINFER_JIT:-0}" = "1" ]; then
      FLASHINFER_ARGS+=( --env "FLASHINFER_DISABLE_JIT_CACHE=1" )
    else
      FLASHINFER_ARGS+=( --env "FLASHINFER_WORKSPACE_DIR=/root/.cache/flashinfer" )
    fi

    # Run container
    exec "$APPT" exec \
      --nv \
      --contain \
      --writable-tmpfs \
      --cleanenv \
      --home "${S}/home:/root" \
      --bind "${S}/hf-cache:/root/.cache/huggingface" \
      --bind "${S}/models:/models" \
      --bind "${S}/cache:/root/.cache" \
      --bind "${S}/tmp:/tmp" \
      --env "HF_HOME=/root/.cache/huggingface" \
      --env "XDG_CACHE_HOME=/root/.cache" \
      --env "TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor" \
      --env "TRITON_CACHE_DIR=/root/.cache/triton" \
      --env "CUDA_CACHE_PATH=/root/.cache/nv" \
      --env "TMPDIR=/tmp" \
      ${HF_TOKEN:+--env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"} \
      --env "CC=cc" --env "CXX=c++" \
      --env "MPICC=" --env "MPICXX=" --env "OMPI_CC=" --env "OMPI_CXX=" \
      "${FLASHINFER_ARGS[@]}" \
      "$SIF" python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --download-dir /models \
        --host 0.0.0.0 --port "${PORT}" \
        --max-model-len "${VLLM_MAX_LEN:-16384}" \
        ${VLLM_TP:+--tensor-parallel-size "${VLLM_TP}"} \
        ${VLLM_ENABLE_LORA:+--enable-lora} \
        ${VLLM_LORAS:+--lora-modules ${VLLM_LORAS}} \
        ${VLLM_QUANT:+--quantization "${VLLM_QUANT}"} \
        ${VLLM_ENFORCE_EAGER:+--enforce-eager} \
        "$@"
  fi

  # Client mode
  PORT=$(<"$PORT_FILE")
  HOST=$(<"$HOST_FILE")
  ENV_HOST="http://${HOST}:${PORT}"

  case "$1" in
    chat)
      shift
      local MSG="${*:-Hello from vLLM on HPC!}"
      curl -sS "${ENV_HOST}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @- <<EOF | jq -r '.choices[0].message.content'
{
  "model": "${MODEL}",
  "messages": [{"role":"user","content":"${MSG}"}],
  "temperature": ${VLLM_TEMP:-0.2},
  "max_tokens": ${VLLM_MAX_TOKENS:-512},
  "stream": false
}
EOF
      ;;
    health)
      curl -sS "${ENV_HOST}/v1/models" | jq
      ;;
    stop)
      pkill -f "python3 -m vllm.entrypoints.openai.api_server" || true
      ;;
    *)
      echo "Forwarding 'vllm $*' to ${ENV_HOST}"
      curl -sS "${ENV_HOST}/$*" || true
      ;;
  esac
}

export -f vllm

