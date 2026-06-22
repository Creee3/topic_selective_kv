# Cloud Experiment Environment

Recorded from the cloud server used to run `topic_selective_kv` experiments.

## Server

- Host prompt: `liuxin@liuxin-1`
- Working directory: `~/working_place/topic_selective_kv`
- Conda environment: `cachegen`
- Recorded command timestamp from `nvidia-smi`: `Thu Jun 4 16:03:16 2026`

## Python

- Python executable: `/home/liuxin/.conda/envs/cachegen/bin/python`
- Python version: `3.10.14`
- Pip executable: `/home/liuxin/.conda/envs/cachegen/bin/pip`
- Pip version: `24.0`

## Core Packages

- `torch`: `2.3.1+cu121`
- PyTorch CUDA runtime: `12.1`
- `transformers`: `4.42.3`
- `numpy`: `1.26.4`
- `fastchat`: import OK
- `sentence-transformers`: `5.5.1`
- `openai`: `0.28.0`

## Relevant `pip freeze`

```text
accelerate==0.32.1
bitsandbytes==0.43.3
# Editable install with no version control (lmcache==0.1)
numpy==1.26.4
openai==0.28.0
sentence-transformers==5.5.1
sentencepiece==0.2.0
torch==2.3.1
torchac==0.9.3
torchac_cuda==0.0.0
transformers==4.42.3
triton==2.3.1
```

## GPU / Driver

- NVIDIA driver: `590.48.01`
- System CUDA version from `nvidia-smi`: `13.1`
- GPUs: `8 x NVIDIA RTX 6000 Ada Generation`
- Per-GPU memory: `49140 MiB`
- At record time, all GPUs showed about `4 MiB` memory usage and no running GPU processes.

### Later GPU Snapshot

From a later `nvidia-smi` snapshot on `Fri Jun 5 06:41:36 2026`:

- Busy GPUs: `0`, `3`, `4`
- Relatively free GPUs: `1`, `2`, `5`, `6`, `7`

This is useful for running multiple single-GPU simulation jobs in parallel with:

```bash
CUDA_VISIBLE_DEVICES=1
CUDA_VISIBLE_DEVICES=2
CUDA_VISIBLE_DEVICES=5
```

and so on.

## Network Proxy

Recorded proxy templates from the cloud environment, with port updated to `7897`.

### IPv4

```bash
export http_proxy=http://172.31.72.144:7897
export https_proxy=http://172.31.72.144:7897
```

### IPv6

```bash
export http_proxy=http://[2001:da8:2d00:2072::1156]:7897
export https_proxy=http://[2001:da8:2d00:2072::1156]:7897
```

## Recommended Experiment Setup

For current single-GPU routing experiments:

```bash
cd ~/working_place/topic_selective_kv
conda activate cachegen

export HF_HOME=~/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export SENTENCE_TRANSFORMERS_HOME=$HF_HOME/sentence_transformers
export TOKENIZERS_PARALLELISM=false
```

If proxy is needed, export either the IPv4 or IPv6 proxy block before running.

## Notes

- Local desktop files are a working/reference copy; cloud results should be treated as the source of truth for experiment outputs.
- Cloud experiments discussed around this record used commands like:

```bash
python distributed_sim.py --model_id ~/models/mistral-7b/ \
    --num_gpus 1 --max_gpu_memory 40 \
    --num_nodes 4 --start_doc 0 --end_doc 200 \
    --passkey --baselines
```
