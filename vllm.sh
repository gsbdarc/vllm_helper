# vllm.sh
#!/usr/bin/env bash

vllm() {
  local LOWERPORT=32768 UPPERPORT=60999
  local S PORT_FILE HOST_FILE PORT HOST ENV_HOST SIF MODEL

  # 1) Require SCRATCH_BASE
  if [ -z "$SCRATCH_BASE" ]; then
    echo "ERROR: please export SCRATCH_BASE" >&2
    return 1
  fi

  # 2) Scratch tree (models, HF cache, tmp, and compile caches)
  S="${SCRATCH_BASE}/vllm"
  mkdir -p "${S}"/{models,hf-cache,tmp,cache}
  PORT_FILE="${S}/port.txt"
  HOST_FILE="${S}/host.txt"

  # 3) Choose your SIF and default model (override via env)
  SIF="${VLLM_SIF:-$PWD/vllm-openai.sif}"
  MODEL="${VLLM_MODEL:-Qwen/Qwen3-14B-Base}"

  # 4) Helper: pick an unused TCP port
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
    # 5) Start server
    PORT=$(find_available_port)
    echo "$PORT" > "$PORT_FILE"
    hostname -s > "$HOST_FILE"

    local BIND="0.0.0.0:${PORT}"
    ENV_HOST="http://$(<"$HOST_FILE"):${PORT}"

    echo "Starting vLLM server binding to ${BIND}"
    echo "Advertising server to clients at ${ENV_HOST}"
    shift

    # Clean CUDA visibility oddities
    unset ROCR_VISIBLE_DEVICES

    # IMPORTANT: --cleanenv to drop host CC/mpicc/etc, then re-add needed env
    exec apptainer exec \
      --nv \
      --contain \
      --cleanenv \
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
      --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
      --env "CC=cc" --env "CXX=c++" \
      --env "MPICC=" --env "MPICXX=" --env "OMPI_CC=" --env "OMPI_CXX=" \
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

  # 6) Client mode: forward subcommands to the running server
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
      # best-effort stop: find the apptainer python server and kill it
      pkill -f "python3 -m vllm.entrypoints.openai.api_server" || true
      ;;
    *)
      echo "Forwarding 'vllm $*' to ${ENV_HOST}"
      curl -sS "${ENV_HOST}/$*" || true
      ;;
  esac
}

export -f vllm

