#!/bin/bash
# Fixed steps for an evolving-ai Part 3 pilot stage P0 pod (speed and repeatability).
# Everything this pod will do is written here at creation time. It accepts one upload
# (the code package, checked against CODE_SHA256), sets up Python and JAX, runs
# scripts/part3/throughput.py, and serves its logs and results read-only on port 8888.
# It accepts no commands. Set at creation: TOKEN, CODE_SHA256, RECV_B64, MACHINE_TAG, GPU (1 or 0).
set -uo pipefail
mkdir -p /w/logs /w/serve
ln -sfn /w/logs /w/serve/logs
log() { echo "$(date -u +%FT%TZ) $*" >> /w/logs/status.txt; }
stop() { log "FAILED: $*"; sleep infinity; }
log boot
export DEBIAN_FRONTEND=noninteractive
{ apt-get update -qq && apt-get install -y -qq python3 git curl ca-certificates; } > /w/logs/apt.log 2>&1 || stop apt
log apt_ok
(cd /w/serve && exec python3 -m http.server 8888 > /w/logs/http.log 2>&1) &
echo "$RECV_B64" | base64 -d > /w/recv.py
log waiting_for_code
python3 /w/recv.py || stop receive
log code_received
mkdir -p /w/code && tar -xzf /w/code.tgz -C /w/code || stop untar
curl -LsSf https://astral.sh/uv/install.sh | sh > /w/logs/uv.log 2>&1 || stop uv_install
export PATH="$HOME/.local/bin:$PATH"
cd /w/code
{ uv python install 3.13 && uv venv --python 3.13 venv3 \
  && uv pip install --python venv3/bin/python -r requirements_part3.txt \
  && uv pip install --python venv3/bin/python --no-deps -e . ; } > /w/logs/setup.log 2>&1 || stop setup
if [ "$GPU" = "1" ]; then
  # the CUDA build of the same JAX version; CUDA libraries come from pip, the driver from the host
  uv pip install --python venv3/bin/python "jax[cuda12]==0.11.2" >> /w/logs/setup.log 2>&1 || stop jax_cuda
fi
git config --global --add safe.directory /w/code
{ venv3/bin/python -c "import sys, jax, jaxlib, flax, optax, evosax, gymnax, rliable; print(sys.version.split()[0], 'jax', jax.__version__, 'backend', jax.default_backend(), jax.devices())"
  echo "commit $(git rev-parse HEAD) dirty_lines $(git status --porcelain -- . ':(exclude)results' | wc -l)"
  nvidia-smi 2>&1 | head -15
  grep -m1 'model name' /proc/cpuinfo; nproc; free -g | head -2; } > /w/logs/env.txt 2>&1
if [ "$GPU" = "1" ]; then
  venv3/bin/python -c "import jax, sys; sys.exit(0 if jax.default_backend() == 'gpu' else 1)" || stop no_gpu_backend
fi
log setup_ok
mkdir -p results/part3/throughput
ln -sfn /w/code/results/part3/throughput /w/serve/throughput
venv3/bin/python scripts/part3/throughput.py --out "results/part3/throughput/$MACHINE_TAG" > /w/logs/throughput.log 2>&1 \
  && log all_done || log "FAILED: throughput (see throughput.log)"
sleep infinity
