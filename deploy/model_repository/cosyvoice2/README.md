# CosyVoice 2 Model Repository

This is a **placeholder** directory. The actual CosyVoice 2 deployment requires 5 separate Triton models that work together:

## Required Models

1. **audio_tokenizer** - Encodes reference audio to speech tokens
2. **speaker_embedding** - Extracts speaker embeddings from reference audio
3. **tensorrt_llm** - TensorRT-LLM optimized text-to-speech LLM
4. **token2wav** - Converts speech tokens to audio waveforms
5. **cosyvoice2** - BLS (Business Logic Script) orchestrator that coordinates all models

## Proper Setup

Run the CosyVoice build script to set up all components:

```bash
./scripts/build_cosyvoice.sh 0 2
```

This will:
1. Clone the CosyVoice repository
2. Download models from HuggingFace and ModelScope
3. Convert the LLM to TensorRT-LLM format
4. Create the proper model repository at `model_repository/cosyvoice2_full/`

## References

- Official Guide: https://github.com/FunAudioLLM/CosyVoice/blob/main/runtime/triton_trtllm/README.md
- Run Script: https://github.com/FunAudioLLM/CosyVoice/blob/main/runtime/triton_trtllm/run.sh

## Performance (L20 GPU)

| Mode | Concurrency | Latency | RTF |
|------|-------------|---------|-----|
| Streaming (first chunk) | 1 | ~190ms | 0.116 |
| Streaming (first chunk) | 4 | ~978ms | 0.073 |
| Offline (full sentence) | 1 | ~758ms | 0.089 |
| Offline (full sentence) | 4 | ~1914ms | 0.061 |
