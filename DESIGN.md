# Real-Time Voice Agent Inference Worker

## Design Document v1.0

---

## 1. Executive Summary

This document outlines the architecture for a high-performance, real-time voice agent inference worker capable of handling concurrent bidirectional voice conversations. The system achieves sub-second latency through GPU-accelerated inference via NVIDIA Triton Inference Server and optimized streaming pipelines.

### Key Goals

- **Latency**: Target end-to-end latency of < 500ms (user silence to first audio response)
- **Concurrency**: Support multiple simultaneous voice sessions per worker
- **Modularity**: Clean separation of concerns with dependency injection for testability
- **Reliability**: Graceful handling of interruptions (barge-in) and connection failures

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              External Clients                               │
└─────────────────────────────────────────────┬───────────────────────────────┘
                                              │ WebSocket (wss://)
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATOR                                    │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐ │
│  │ Connection Mgr  │  │  Session Router  │  │    Tool Request Handler     │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────┬───────────────────────────────┘
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────┐
        │                                     │                             │
        ▼                                     ▼                             ▼
┌───────────────────┐               ┌─────────────────┐            ┌───────────────────┐
│     ASR RAIL      │               │    LLM RAIL     │            │     TTS RAIL      │
│  ┌─────────────┐  │               │                 │            │  ┌─────────────┐  │
│  │ Silero VAD  │  │  user_text    │  ┌───────────┐  │ agent_text │  │ CosyVoice 2 │  │
│  │  (1.8 MB)   │──┼──────────────►│  │  Qwen 3   │  │───────────►│  │  (0.5B)     │  │
│  └─────────────┘  │               │  │   4B      │  │            │  └─────────────┘  │
│  ┌─────────────┐  │               │  │  INT4     │  │            │        │         │
│  │ Parakeet    │  │               │  └───────────┘  │            │        ▼         │
│  │ TDT 0.6B   │  │               │        │        │            │   audio_out      │
│  └─────────────┘  │               │        ▼        │            │        │         │
│        │         │               │  tool_request   │            │        │         │
│        ▼         │               │        │        │            │   ┌────▼────┐    │
│  user_speaking   │               └────────┼────────┘            │   │Interrupt│    │
│  (VAD signal)────┼───────────────────────────────────────────────►  │ Handler │    │
└───────────────────┘               ◄───────┘                      │   └─────────┘    │
                                tool_response                      └───────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRITON INFERENCE SERVER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Silero VAD  │  │  Parakeet    │  │   Qwen 3     │  │   Chatterbox     │ │
│  │    (ONNX)    │  │  TDT 0.6B    │  │  4B AWQ      │  │   TTS (PyTorch)  │ │
│  │              │  │ (ONNX RT)    │  │   (vLLM)     │  │                  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Design

### 3.1 Orchestrator

The Orchestrator is the central coordination layer that manages client connections and routes data between rails.

#### Responsibilities

- Accept and manage WebSocket connections
- Create and destroy `AgentRail` instances per session
- Route audio/text between components
- Handle tool requests/responses from external systems
- Manage session lifecycle and state

#### WebSocket Protocol

```
Client → Server Messages:
├── audio_chunk     : Binary audio data (16kHz, mono, PCM16)
├── tool_response   : JSON { request_id, result }
└── control         : JSON { action: "hangup" | "mute" | "unmute" }

Server → Client Messages:
├── audio_chunk     : Binary audio data (22.05kHz, mono, PCM16)
├── transcript      : JSON { text, is_final, confidence }
├── tool_request    : JSON { request_id, tool_name, parameters }
├── agent_state     : JSON { speaking: bool, processing: bool }
└── error           : JSON { code, message }
```

#### Connection Manager

```python
class ConnectionManager:
    """Manages concurrent WebSocket sessions."""

    async def accept(self, websocket: WebSocket) -> str:
        """Accept connection, return session_id."""

    async def disconnect(self, session_id: str) -> None:
        """Clean up session resources."""

    def get_session(self, session_id: str) -> AgentRail:
        """Retrieve active session."""
```

### 3.2 AgentRail

A stateful container created for each active voice session, coordinating the three inference rails.

```python
class AgentRail:
    """Per-session voice agent pipeline."""

    def __init__(
        self,
        asr_rail: ASRRail,
        llm_rail: LLMRail,
        tts_rail: TTSRail,
        triton_client: TritonClient,
    ):
        self.asr = asr_rail
        self.llm = llm_rail
        self.tts = tts_rail
        self.conversation_history: list[Message] = []
        self.is_user_speaking: bool = False

    async def process_audio(self, audio_chunk: bytes) -> AsyncIterator[bytes]:
        """Main processing pipeline."""
```

### 3.3 ASRRail (Automatic Speech Recognition)

Converts streaming audio input to text while detecting voice activity.

#### Components

| Component | Model | Purpose | Latency |
|-----------|-------|---------|---------|
| VAD | Silero VAD v5 | Voice activity detection | < 1ms per 30ms chunk |
| ASR | Parakeet TDT 0.6B V2 | Speech-to-text | RTFx 3380 |

#### Parakeet TDT 0.6B V2 Specifications

- **Architecture**: FastConformer encoder + TDT (Token-and-Duration Transducer) decoder
- **Parameters**: 600 million
- **Training Data**: ~120,000 hours (Granary dataset)
- **Input**: 16kHz mono audio
- **Max Duration**: Up to 24 minutes per segment
- **Features**: Automatic punctuation, capitalization, word-level timestamps
- **WER**: 6.05% average (Hugging Face Open ASR Leaderboard leader)
- **License**: CC-BY-4.0 (commercial use allowed)

#### Silero VAD Specifications

- **Size**: 1.8 MB (quantized ONNX)
- **Sample Rates**: 8kHz, 16kHz
- **Chunk Size**: 30ms minimum
- **Processing Time**: < 1ms per chunk on single CPU thread
- **Languages**: Trained on 6000+ languages
- **License**: MIT

#### Interface

```python
class ASRRail:
    """Streaming ASR with voice activity detection."""

    async def feed_audio(self, chunk: bytes) -> AsyncIterator[ASREvent]:
        """
        Process audio chunk, yield events.

        Events:
        - VADEvent(is_speech: bool, confidence: float)
        - TranscriptEvent(text: str, is_final: bool, timestamps: list)
        """

    @property
    def is_user_speaking(self) -> bool:
        """Current VAD state."""

    def reset(self) -> None:
        """Clear buffers for new utterance."""
```

#### VAD State Machine

```
                    ┌──────────────────┐
         speech    │                  │    silence (>300ms)
    ┌─────────────►│  USER_SPEAKING   ├──────────────────┐
    │              │                  │                  │
    │              └──────────────────┘                  │
    │                                                    ▼
┌───┴──────────────┐                        ┌──────────────────┐
│                  │       silence          │                  │
│     IDLE         │◄───────────────────────│  END_OF_SPEECH   │
│                  │       (>1000ms)        │                  │
└──────────────────┘                        └──────────────────┘
```

### 3.4 LLMRail (Language Model)

Processes transcribed user text and generates agent responses with tool-calling capability.

#### Qwen3 4B Specifications

- **Architecture**: Dense transformer with GQA (32 Q heads, 8 KV heads)
- **Parameters**: 4 billion (3.6B non-embedding)
- **Quantization**: AWQ INT4 (W4A16)
- **Context Window**: 262,144 tokens (limited to 8,192 for memory efficiency)
- **Features**: Instruction following, reasoning, function calling
- **Inference**: vLLM with KV cache + in-flight batching

#### Performance Optimizations

| Optimization | Impact |
|--------------|--------|
| AWQ INT4 | ~4x memory reduction vs FP16 |
| KV Cache | Eliminates redundant computation |
| In-flight Batching | Maximizes GPU utilization |
| Streaming Output | Reduces time-to-first-token |

#### Interface

```python
class LLMRail:
    """Streaming LLM with tool-calling support."""

    async def generate(
        self,
        user_text: str,
        conversation_history: list[Message],
        tools: list[ToolDefinition],
    ) -> AsyncIterator[LLMEvent]:
        """
        Generate response, yielding events.

        Events:
        - TextDeltaEvent(text: str)
        - ToolCallEvent(request_id: str, tool_name: str, parameters: dict)
        - CompletionEvent(full_text: str, usage: TokenUsage)
        """

    async def submit_tool_response(
        self,
        request_id: str,
        result: Any,
    ) -> AsyncIterator[LLMEvent]:
        """Continue generation after tool execution."""

    def interrupt(self) -> None:
        """Cancel ongoing generation (barge-in)."""
```

#### Tool Calling Protocol

```python
# Tool Definition
class ToolDefinition:
    name: str
    description: str
    parameters: JSONSchema

# Example Tool
{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}
```

### 3.5 TTSRail (Text-to-Speech)

Converts agent text responses to streaming audio output.

#### Chatterbox TTS Specifications

- **Architecture**: T3 speech token LLM + S3Gen flow decoder
- **T3 Backend**: vLLM for speech token generation
- **S3Gen Backend**: PyTorch with torch.compile()
- **First Packet Latency**: ~200ms (streaming mode)
- **Features**: Voice cloning, emotion control
- **Optimizations**: Progressive streaming with variable chunk sizes

#### Interface

```python
class TTSRail:
    """Streaming TTS with interruption support."""

    async def synthesize(
        self,
        text_stream: AsyncIterator[str],
        voice_id: str = "default",
    ) -> AsyncIterator[bytes]:
        """
        Convert streaming text to streaming audio.

        Yields: PCM16 audio chunks (22.05kHz, mono)
        """

    def interrupt(self) -> None:
        """Stop synthesis immediately (barge-in)."""

    @property
    def is_speaking(self) -> bool:
        """Whether TTS is actively outputting."""
```

#### Barge-In Handling

When `ASRRail.is_user_speaking` becomes `True`:

1. `TTSRail.interrupt()` is called immediately
2. Audio output buffer is flushed
3. `LLMRail.interrupt()` cancels ongoing generation
4. System enters listening mode

---

## 4. Data Flow

### 4.1 Happy Path (User → Agent → User)

```
Time →
────────────────────────────────────────────────────────────────────►

User speaks          VAD detects       ASR transcribes      LLM generates
[Audio In]  ───────► [Speech Start] ──► [Text]  ──────────► [Response]
                                                                  │
                                                                  ▼
User hears           Audio sent         TTS synthesizes      [Stream]
[Audio Out] ◄─────── [WebSocket]  ◄──── [Audio] ◄─────────── [Text]
```

### 4.2 Barge-In Flow

```
Agent speaking...    User interrupts    Immediate stop
[TTS Active] ────────► [VAD: Speech] ──► [TTS.interrupt()]
                                              │
                                              ▼
                                        [LLM.interrupt()]
                                              │
                                              ▼
                                        [Listen Mode]
```

### 4.3 Tool Calling Flow

```
LLM needs data         Tool emitted        External call
[Generation] ──────────► [tool_request] ──► [Client handles]
                                                  │
                                                  ▼
LLM continues          Response received   [tool_response]
[Streaming] ◄────────── [submit_tool_response] ◄──┘
```

---

## 5. Triton Inference Server Configuration

### 5.1 Model Repository Structure

```
model_repository/
├── silero_vad/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
├── parakeet_tdt/              # ASR orchestrator (Python BLS)
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
├── parakeet_encoder/          # ONNX Runtime backend
│   └── 1/model.py
├── parakeet_decoder/          # ONNX Runtime backend
│   └── 1/model.py
├── qwen3/
│   ├── config.pbtxt
│   └── 1/
│       └── model.json          # vLLM config
├── t3/                         # Speech token generator (vLLM)
│   └── 1/model.json
└── chatterbox/                 # TTS orchestrator (Python BLS)
    ├── config.pbtxt
    └── 1/
        └── model.py
```

### 5.2 Instance Configuration

```protobuf
# Example: qwen3/config.pbtxt
name: "qwen3"
backend: "vllm"
max_batch_size: 8

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]

dynamic_batching {
  preferred_batch_size: [1, 2, 4, 8]
  max_queue_delay_microseconds: 100000
}

parameters {
  key: "gpt_model_type"
  value: { string_value: "inflight_fused_batching" }
}

parameters {
  key: "max_tokens_in_paged_kv_cache"
  value: { string_value: "8192" }
}
```

### 5.3 Sequence Batching for Stateful Models

For models that maintain state across requests (like streaming ASR):

```protobuf
sequence_batching {
  max_sequence_idle_microseconds: 5000000
  control_input [
    {
      name: "START"
      control [
        { kind: CONTROL_SEQUENCE_START }
      ]
    },
    {
      name: "END"
      control [
        { kind: CONTROL_SEQUENCE_END }
      ]
    }
  ]
}
```

---

## 6. Concurrency Model

### 6.1 Per-Request Concurrency

```python
# Each WebSocket connection runs independently
async def handle_session(websocket: WebSocket, session_id: str):
    rail = AgentRail(...)

    async def audio_input_task():
        async for chunk in websocket.iter_bytes():
            async for event in rail.asr.feed_audio(chunk):
                await process_asr_event(event)

    async def audio_output_task():
        async for chunk in rail.tts.output_stream():
            await websocket.send_bytes(chunk)

    await asyncio.gather(audio_input_task(), audio_output_task())
```

### 6.2 GPU Resource Management

| Model | GPU Memory | Instances | Concurrency |
|-------|------------|-----------|-------------|
| Silero VAD | ~50 MB | CPU only | Unlimited |
| Parakeet TDT 0.6B | ~2 GB | 1 | Batched (8) |
| Qwen3 4B INT4 | ~3 GB | 1 | In-flight batching |
| CosyVoice 2 | ~2 GB | 1 | Batched (4) |

**Total GPU Memory**: ~7 GB (fits on single 8GB+ GPU)

### 6.3 Backpressure Handling

```python
class AudioBuffer:
    """Bounded buffer with backpressure."""

    def __init__(self, max_duration_ms: int = 5000):
        self.max_chunks = max_duration_ms // 20  # 20ms chunks
        self.buffer: asyncio.Queue = asyncio.Queue(maxsize=self.max_chunks)

    async def put(self, chunk: bytes) -> None:
        try:
            self.buffer.put_nowait(chunk)
        except asyncio.QueueFull:
            # Drop oldest to maintain real-time
            self.buffer.get_nowait()
            await self.buffer.put(chunk)
```

---

## 7. Latency Budget

Target: **< 500ms** end-to-end (user silence → first audio byte)

| Component | Target | Notes |
|-----------|--------|-------|
| Network (WebSocket) | 20-50ms | Depends on client location |
| VAD Detection | < 5ms | Silero on CPU |
| ASR Inference | 50-100ms | Parakeet streaming |
| LLM TTFT | 150-200ms | Qwen3 4B with batching |
| TTS First Byte | 100-150ms | CosyVoice 2 streaming |
| **Total** | **325-505ms** | Within budget |

### Latency Optimization Strategies

1. **Regional Colocation**: All components in same VPC/region
2. **Persistent Connections**: Reuse gRPC channels to Triton
3. **Streaming Everything**: No waiting for complete utterances
4. **Speculative Execution**: Start TTS before LLM completes
5. **Audio Chunking**: 20ms chunks for minimal buffering delay

---

## 8. API Design

### 8.1 REST Endpoints

```
POST   /v1/sessions              Create new voice session
DELETE /v1/sessions/{id}         End session
GET    /v1/sessions/{id}/status  Get session state
POST   /v1/sessions/{id}/tools   Submit tool response
```

### 8.2 WebSocket Endpoint

```
WS /v1/sessions/{id}/stream

# Connection lifecycle:
1. Client connects with session_id
2. Server sends: {"type": "connected", "session_id": "..."}
3. Client streams audio, server streams audio + events
4. Either side can close cleanly
```

### 8.3 Message Types

```typescript
// Client → Server
type ClientMessage =
  | { type: "audio"; data: ArrayBuffer }
  | { type: "tool_response"; request_id: string; result: any }
  | { type: "control"; action: "mute" | "unmute" | "hangup" }

// Server → Client
type ServerMessage =
  | { type: "audio"; data: ArrayBuffer }
  | { type: "transcript"; text: string; is_final: boolean }
  | { type: "tool_request"; request_id: string; tool: string; params: any }
  | { type: "state"; agent_speaking: boolean; processing: boolean }
  | { type: "error"; code: string; message: string }
```

---

## 9. Deployment Architecture

### 9.1 Single-Node Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                      EC2 Instance                            │
│                   (g5.2xlarge / A10G)                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Docker Compose                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │ │
│  │  │   Nginx     │  │   Worker    │  │ Triton Server   │  │ │
│  │  │  (TLS/WS)   │──│  (Python)   │──│   (GPU)         │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Multi-Node Scaling

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (WebSocket    │
                    │    sticky)      │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │ Worker  │         │ Worker  │         │ Worker  │
    │   +     │         │   +     │         │   +     │
    │ Triton  │         │ Triton  │         │ Triton  │
    └─────────┘         └─────────┘         └─────────┘
```

### 9.3 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA T4 (16GB) | NVIDIA A10G (24GB) |
| CPU | 8 cores | 16 cores |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB SSD | 200 GB NVMe |
| Network | 1 Gbps | 10 Gbps |

---

## 10. Monitoring & Observability

### 10.1 Key Metrics

```python
# Latency metrics (histograms)
voice_agent_asr_latency_seconds
voice_agent_llm_ttft_seconds
voice_agent_llm_total_seconds
voice_agent_tts_ttfb_seconds
voice_agent_e2e_latency_seconds

# Throughput metrics (counters)
voice_agent_sessions_total
voice_agent_audio_bytes_received_total
voice_agent_audio_bytes_sent_total
voice_agent_tokens_generated_total

# Error metrics (counters)
voice_agent_errors_total{type="asr|llm|tts|connection"}

# Resource metrics (gauges)
voice_agent_active_sessions
voice_agent_gpu_memory_used_bytes
triton_inference_queue_duration_us
```

### 10.2 Health Checks

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "triton": await check_triton(),
        "active_sessions": session_manager.count(),
        "gpu_memory_free": get_gpu_memory_free(),
    }

@app.get("/ready")
async def ready():
    # Only ready if all models loaded
    return {"ready": all_models_loaded()}
```

### 10.3 Distributed Tracing

Each request carries a trace context through the pipeline:

```
[WebSocket] → [ASR] → [LLM] → [TTS] → [WebSocket]
    │           │        │       │         │
    └───────────┴────────┴───────┴─────────┘
                    Trace ID
```

---

## 11. Error Handling

### 11.1 Error Categories

| Category | Examples | Recovery |
|----------|----------|----------|
| Transient | Triton timeout, GPU OOM | Retry with backoff |
| Connection | WebSocket disconnect | Clean up session |
| Model | Invalid input, generation failure | Return error to client |
| System | Disk full, service crash | Alert, restart |

### 11.2 Graceful Degradation

```python
async def process_with_fallback(audio: bytes) -> str:
    try:
        return await asr_rail.transcribe(audio, timeout=5.0)
    except TimeoutError:
        # Fallback to smaller/faster model or cached response
        logger.warning("ASR timeout, using fallback")
        return await asr_fallback.transcribe(audio)
```

---

## 12. Security Considerations

### 12.1 Connection Security

- TLS 1.3 for all WebSocket connections
- Authentication via JWT or API key in connection handshake
- Rate limiting per client IP

### 12.2 Data Privacy

- Audio data processed in-memory only (not persisted by default)
- Conversation history cleared on session end
- Option to disable logging of transcripts

### 12.3 Input Validation

```python
def validate_audio_chunk(data: bytes) -> bool:
    # Max chunk size: 1 second of 16kHz PCM16 = 32KB
    if len(data) > 32768:
        raise ValueError("Audio chunk too large")
    if len(data) % 2 != 0:  # PCM16 = 2 bytes per sample
        raise ValueError("Invalid PCM16 data")
    return True
```

---

## 13. Testing Strategy

### 13.1 Unit Tests

- Individual rail components with mocked Triton client
- Audio processing utilities
- Protocol message serialization

### 13.2 Integration Tests

- Full pipeline with test audio files
- Tool calling flow
- Barge-in scenarios

### 13.3 Load Tests

- Concurrent session handling
- GPU memory under load
- Latency percentiles (P50, P95, P99)

### 13.4 Chaos Tests

- Triton server restart during inference
- Network partition simulation
- GPU memory exhaustion

---

## 14. Future Enhancements

1. **Multi-language Support**: Extend ASR/TTS to support additional languages
2. **Voice Cloning**: Custom voice profiles per session
3. **Emotion Detection**: Analyze user sentiment from audio
4. **Multi-modal**: Add vision capabilities for video calls
5. **Edge Deployment**: Optimize for NVIDIA Jetson devices

---

## References

- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
- [Parakeet TDT 0.6B V2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Qwen3 TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md)
- [Sub-Second Voice Agent Latency Guide](https://dev.to/tigranbs/sub-second-voice-agent-latency-a-practical-architecture-guide-4cg1)
