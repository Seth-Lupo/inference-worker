# Deploy

Triton Inference Server deployment for voice agent.

## Quick Start

```bash
# 1. Setup EC2 instance (installs Docker, nvidia-container-toolkit)
./scripts/setup_ec2.sh

# 2. Build all models
./scripts/build_all.sh

# 3. Start services
docker compose up -d
```

## Directory Structure

```
deploy/
├── docker-compose.yml      # Main orchestration (triton + worker)
├── Dockerfile.triton       # Triton image with Python deps
├── Dockerfile.worker       # Voice agent worker image
├── .env.example            # Environment template (copy to .env)
│
├── scripts/
│   ├── common.sh           # Shared utilities (logging, LFS, docker)
│   │
│   │  # Build scripts
│   ├── build_all.sh        # Build all models in sequence
│   ├── build_cosyvoice.sh  # CosyVoice TTS (7 Triton models)
│   ├── build_qwen3.sh      # Qwen3 LLM (TensorRT-LLM)
│   ├── build_parakeet.sh   # Parakeet ASR (ONNX)
│   ├── build_parakeet_trt.sh # (Optional) Parakeet TensorRT engines
│   ├── clean_models.sh     # Cleanup build artifacts
│   │
│   │  # Container management
│   ├── setup_ec2.sh        # EC2 instance setup
│   ├── deploy.sh           # Full deploy (build + start)
│   ├── start.sh            # Start containers
│   ├── stop.sh             # Stop containers
│   ├── status.sh           # Container status
│   ├── logs.sh             # View logs
│   │
│   │  # Debugging (optional)
│   ├── audit_cosyvoice_gpu.sh  # Check GPU config
│   └── fix_cosyvoice_gpu.sh    # Fix GPU config locally
│
├── model_repository/       # Triton models (created by build scripts)
│   ├── parakeet_tdt/       # ASR model
│   ├── qwen3/              # LLM model (vLLM)
│   └── cosyvoice2_full/    # TTS models (7 submodels)
│
├── cosyvoice_build/        # CosyVoice build artifacts
│   ├── CosyVoice/          # Cloned repo (mounted for Python imports)
│   ├── CosyVoice2-0.5B/    # ModelScope weights
│   ├── cosyvoice2_llm/     # HuggingFace LLM
│   └── trt_engines_*/      # Built TensorRT engines
│
└── qwen3_build/            # Qwen3 build artifacts
    └── engine_int4/        # Built TensorRT-LLM engine
```

## Build Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `build_cosyvoice.sh -1 2` | Clone + download + build TTS | ~20 min |
| `build_qwen3.sh` | Download + build LLM | ~30 min |
| `build_parakeet.sh` | Download ASR model | ~5 min |
| `build_all.sh` | All of the above | ~60 min |

### CosyVoice Stages

```bash
./scripts/build_cosyvoice.sh -1 2   # All stages
./scripts/build_cosyvoice.sh -1 -1  # Clone only (re-clones)
./scripts/build_cosyvoice.sh 0 0    # Download models only
./scripts/build_cosyvoice.sh 1 1    # Build TRT engines only
./scripts/build_cosyvoice.sh 2 2    # Create model repo only
./scripts/build_cosyvoice.sh cleanup --all  # Remove everything
```

## Models

| Model | Type | Backend | GPU |
|-------|------|---------|-----|
| `qwen3` | LLM | vLLM | Yes |
| `parakeet_tdt` | ASR | Python (PyTorch) | Yes |
| `cosyvoice2` | TTS Orchestrator | Python (BLS) | Yes |
| `tensorrt_llm` | TTS LLM | TensorRT-LLM | Yes |
| `audio_tokenizer` | TTS | Python | Yes |
| `speaker_embedding` | TTS | Python | Yes |
| `token2wav` | TTS Vocoder | Python | Yes |

## Environment Variables

```bash
# .env file
HF_TOKEN=hf_xxx           # HuggingFace token (required for gated models)
CUDA_VISIBLE_DEVICES=0    # GPU selection
```

## Troubleshooting

### LFS files not downloaded
```bash
cd deploy/cosyvoice_build/CosyVoice2-0.5B
git lfs pull
```

### Rebuild model repository
```bash
./scripts/build_cosyvoice.sh 2 2
```

### Check model status
```bash
curl localhost:8000/v2/models | jq
```
