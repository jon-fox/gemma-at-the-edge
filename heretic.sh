#!/bin/bash
# Usage: HF_TOKEN=hf_xxx bash heretic_setup.sh
# apt-get update -qq && apt-get install -y vim
set -e

apt-get update -qq && apt-get install -y -q vim

pip uninstall -y torchvision torchaudio heretic-llm
pip install -q accelerate==1.10.0 torch==2.8.0 triton==3.4.0 huggingface_hub
pip install -q git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
pip install -q git+https://github.com/huggingface/transformers.git
pip install -q git+https://github.com/p-e-w/heretic.git@ara

# heretic --model openai/gpt-oss-20b --trust-remote-code true --device-map cuda:0

# Check if abliterated model already exists on HuggingFace
if python -c "from huggingface_hub import repo_exists; exit(0 if repo_exists('foxj77/gpt-oss-20b-heretic', token='$HF_TOKEN') else 1)" 2>/dev/null; then
    echo "Found foxj77/gpt-oss-20b-heretic on HuggingFace, downloading..."
    heretic --model foxj77/gpt-oss-20b-heretic --trust-remote-code true --device-map cuda:0 --dtypes '["bfloat16"]'
else
    echo "No existing model found, running abliteration..."
    heretic --model openai/gpt-oss-20b --trust-remote-code true --device-map cuda:0 --dtypes '["bfloat16"]'
fi