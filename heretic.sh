#!/bin/bash
# Usage: HF_TOKEN=hf_xxx bash heretic_setup.sh
# apt-get update -qq && apt-get install -y vim

set -e

apt-get update -qq && apt-get install -y -q vim

export PIP_ROOT_USER_ACTION=ignore

# RunPod wipes / on pod restart but keeps /workspace, so cache the model there.
# This makes the ~40GB download a one-time cost across pods.
export HF_HOME=/workspace/hf-cache
export HF_XET_HIGH_PERFORMANCE=1

# Only uninstall what's actually there
for pkg in torchvision torchaudio heretic-llm; do
    if pip show "$pkg" >/dev/null 2>&1; then
        echo "Removing $pkg..."
        pip uninstall -y -q "$pkg"
    fi
done

# A package installed by apt has no RECORD file, so pip can't upgrade over it and
# hard-fails with uninstall-no-record-file. Only shadow it when that's the case.
if pip show cryptography >/dev/null 2>&1 \
   && ! python -c "import importlib.metadata as md; md.distribution('cryptography').read_text('RECORD') or exit(1)" 2>/dev/null; then
    echo "cryptography is apt-managed, shadowing with a pip copy..."
    pip install -q --ignore-installed cryptography
fi

pip install -q accelerate==1.10.0 torch==2.8.0 triton==3.4.0 huggingface_hub
pip install -q git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
pip install -q git+https://github.com/huggingface/transformers.git
pip install -q git+https://github.com/p-e-w/heretic.git@ara

# Fetch weights up front rather than letting heretic do it, so a stalled
# transfer can be retried without restarting the whole run. Downloads resume
# from whatever is already in the cache.
fetch() {
    local repo=$1
    for attempt in 1 2 3 4 5; do
        echo "Downloading $repo (attempt $attempt)..."
        hf download "$repo" && return 0
        echo "Stalled or failed, retrying in 5s..."
        sleep 5
        # Xet's chunk reconstruction is what tends to hang; drop to plain HTTP.
        export HF_HUB_DISABLE_XET=1
    done
    echo "Could not download $repo after 5 attempts" >&2
    return 1
}

# Check if abliterated model already exists on HuggingFace
if python -c "from huggingface_hub import repo_exists; exit(0 if repo_exists('foxj77/gpt-oss-20b-heretic', token='$HF_TOKEN') else 1)" 2>/dev/null; then
    echo "Found foxj77/gpt-oss-20b-heretic on HuggingFace, downloading..."
    fetch foxj77/gpt-oss-20b-heretic
    heretic --model foxj77/gpt-oss-20b-heretic --trust-remote-code true --device-map cuda:0 --dtypes '["bfloat16"]'
else
    echo "No existing model found, running abliteration..."
    fetch openai/gpt-oss-20b
    heretic --model openai/gpt-oss-20b --trust-remote-code true --device-map cuda:0 --dtypes '["bfloat16"]'
fi

