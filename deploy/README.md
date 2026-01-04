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
│   ├── build_chatterbox.sh # Chatterbox TTS (T3 + S3Gen)
│   ├── build_qwen.sh       # Qwen3 LLM (vLLM)
│   ├── build_parakeet.sh   # Parakeet ASR (ONNX Runtime)
│   ├── clean_models.sh     # Cleanup build artifacts
│   │
│   │  # Container management
│   ├── setup_ec2.sh        # EC2 instance setup
│   ├── deploy.sh           # Full deploy (build + start)
│   ├── start.sh            # Start containers
│   ├── stop.sh             # Stop containers
│   ├── status.sh           # Container status
│   └── logs.sh             # View logs
│
├── model_repository/       # Triton models (created by build scripts)
│   ├── llm/
│   │   ├── qwen3/          # LLM (vLLM backend)
│   │   └── t3/             # T3 speech tokens (vLLM backend)
│   ├── tts/
│   │   ├── chatterbox/     # TTS orchestrator (Python BLS)
│   │   ├── chatterbox_voice_encoder/  # Voice cloning (Python)
│   │   ├── chatterbox_s3gen/          # Audio synthesis (Python + torch.compile)
│   │   └── chatterbox_assets/         # Model weights
│   └── asr/
│       ├── parakeet_tdt/   # ASR orchestrator (Python BLS)
│       ├── parakeet_encoder/  # Encoder (ONNX Runtime)
│       ├── parakeet_decoder/  # Decoder (ONNX Runtime)
│       └── parakeet_onnx/     # ONNX model files
│
├── models/                 # Downloaded model weights
│   ├── t3_weights/         # T3 vLLM weights
│   └── qwen3_weights/      # Qwen3 vLLM weights
│
└── chatterbox_build/       # Chatterbox build artifacts
```

## Build Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `build_chatterbox.sh` | Download + setup Chatterbox TTS | ~10 min |
| `build_qwen.sh` | Download Qwen3 LLM weights | ~5 min |
| `build_parakeet.sh` | Download Parakeet ASR ONNX | ~5 min |
| `build_all.sh` | All of the above | ~20 min |

## Models

| Model | Type | Backend | GPU |
|-------|------|---------|-----|
| `qwen3` | LLM | vLLM | Yes |
| `t3` | Speech Tokens | vLLM | Yes |
| `chatterbox` | TTS Orchestrator | Python (BLS) | Yes |
| `chatterbox_voice_encoder` | Voice Cloning | Python (PyTorch) | Yes |
| `chatterbox_s3gen` | Audio Synthesis | Python (torch.compile) | Yes |
| `parakeet_tdt` | ASR Orchestrator | Python (BLS) | Yes |
| `parakeet_encoder` | ASR Encoder | Python (ONNX Runtime) | Yes |
| `parakeet_decoder` | ASR Decoder | Python (ONNX Runtime) | Yes |

## Environment Variables

```bash
# .env file
HF_TOKEN=hf_xxx           # HuggingFace token (required for gated models)
CUDA_VISIBLE_DEVICES=0    # GPU selection
```

## Troubleshooting

### Check model status
```bash
curl localhost:8000/v2/models | jq
```

### View Triton logs
```bash
docker compose logs -f triton
```

### Rebuild specific model
```bash
./scripts/build_chatterbox.sh cleanup --all
./scripts/build_chatterbox.sh
```
