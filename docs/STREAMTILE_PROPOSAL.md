# StreamTile: Layer-wise Weight Streaming for cutileGPT

> **"Don't fit the model to your GPU. Stream it."**
>
> *모델을 GPU에 맞추지 마라. 스트리밍하라.*

## Executive Summary

StreamTile은 cutileGPT의 역발상 철학을 메모리 관리에 적용한 아키텍처입니다. 모델 압축 대신 **가중치 스트리밍**을 통해 제한된 GPU 메모리로 대형 모델을 실행합니다.

---

## 영감: SpaceX의 역발상

| SpaceX 철학 | StreamTile 철학 |
|-------------|-----------------|
| 로켓 재사용 | 가중치 스트리밍/재사용 |
| 발사당 비용 ↓ | 추론당 메모리 ↓ |
| N회 발사 가능 | 더 큰 모델 실행 가능 |
| 간단한 재료 (스테인레스) | 선언적 프로그래밍 (Tile) |

---

## 현재 문제: GPU 메모리 병목

```
GPT-2 (117M) = ~500MB (FP32)
GPT-2 XL (1.5B) = ~6GB (FP32)
LLaMA-7B = ~28GB (FP32)

소비자 GPU: 8-24GB
→ 전체 모델 상주 = 메모리 낭비 + 느린 콜드 스타트
```

### 기존 접근법의 한계

```
기존 방식: 모델을 줄여라!
├── Distillation (지식 증류)
├── Quantization (INT8, INT4)
├── Pruning (가지치기)
└── 결과: 모델은 작아졌지만...
    ├── 프레임워크는 2GB (PyTorch)
    ├── 복잡한 최적화 파이프라인
    ├── 전문가만 할 수 있음
    └── 모델마다 새로 최적화 필요
```

---

## 제안: StreamTile 아키텍처

### 핵심 아이디어

```
기존 방식: 모든 레이어를 GPU에 로드 → 추론
StreamTile: 필요한 레이어만 로드 → 추론 → 언로드/캐시

전통적 프로그래밍 비유:
- 정적 링킹 (Static Linking): 모든 라이브러리를 실행파일에 포함
- 동적 링킹 (Dynamic Linking): 필요할 때만 .so/.dll 로드

StreamTile = 가중치의 "동적 링킹"
```

### 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                    Weight Store (CPU/NVMe SSD)                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐            │
│  │ Layer 0 │ │ Layer 1 │ │ Layer 2 │ ... │ Layer N │            │
│  │  ~40MB  │ │  ~40MB  │ │  ~40MB  │     │  ~40MB  │            │
│  └─────────┘ └─────────┘ └─────────┘     └─────────┘            │
└─────────────────────────────────────────────────────────────────┘
        │              │                         │
        │   Prefetch   │  On-demand              │
        ▼              ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Weight Cache (GPU Memory)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Layer i   │  │  Layer i+1  │  │   (Pool)    │              │
│  │  (Active)   │  │ (Prefetch)  │  │   (LRU)     │              │
│  │  Computing  │  │  Loading    │  │  Evictable  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────┐                   │
│  │     Tile Programming Kernels             │                   │
│  │     cutile_attention, cutile_mlp, ...    │                   │
│  └──────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tile Programming이 이 방식에 적합한 이유

### 1. 선언적 데이터 명세

```python
# 현재 Tile Programming
@ct.kernel
def attention(Q, K, V, Out, ...):
    q = ct.load(Q, ...)  # 컴파일러가 Q 필요함을 알음
    k = ct.load(K, ...)  # 컴파일러가 K 필요함을 알음
    # ... 계산 ...

# StreamTile 확장 (제안)
@ct.kernel(streaming=True)
def attention(Q, K, V, Out, ...):
    # 컴파일러가 자동으로:
    # 1. Q, K, V 프리페치 스케줄링
    # 2. 계산 중 다음 레이어 프리로드
    # 3. 완료 후 필요없는 텐서 해제
```

### 2. TMA (Tensor Memory Accelerator) 활용

```python
# Hopper/Blackwell의 TMA는 비동기 메모리 전송 지원
# Tile Programming이 이를 자동 활용 가능

class StreamingWeightManager:
    """TMA를 활용한 비동기 가중치 스트리밍"""

    def prefetch_async(self, layer_idx: int):
        """다음 레이어 비동기 프리페치"""
        # TMA로 CPU→GPU 비동기 전송
        # 현재 계산과 오버랩

    def get_or_load(self, layer_idx: int) -> WeightTile:
        """캐시에 있으면 반환, 없으면 로드"""
        if layer_idx in self.cache:
            return self.cache[layer_idx]
        return self._load_and_cache(layer_idx)
```

---

## 구체적 구현 제안

### 1. WeightStore: 가중치 저장소

```python
class WeightStore:
    """
    가중치의 "라이브러리" - 필요시 로드
    mmap 스타일로 파일에서 직접 로드 가능
    """

    def __init__(self, model_path: str):
        self.path = model_path
        self.layer_offsets = self._parse_index()

    def load_layer(self, layer_idx: int) -> Dict[str, cp.ndarray]:
        """특정 레이어만 GPU로 로드"""
        offset, size = self.layer_offsets[layer_idx]

        # Zero-copy: mmap + cupy DLPack
        with self._mmap_region(offset, size) as data:
            return self._to_gpu(data)

    def stream_layer_async(self, layer_idx: int, stream: cp.cuda.Stream):
        """비동기 스트리밍 로드"""
        # CUDA stream으로 오버랩
```

### 2. LayerCache: LRU 캐시

```python
class LayerCache:
    """
    GPU 메모리의 "페이지 캐시"
    자주 쓰는 레이어 유지, 안 쓰는 레이어 제거
    """

    def __init__(self, max_layers: int = 4):
        self.max_layers = max_layers
        self.cache = OrderedDict()  # LRU

    def get(self, layer_idx: int) -> Optional[LayerWeights]:
        if layer_idx in self.cache:
            self.cache.move_to_end(layer_idx)  # LRU update
            return self.cache[layer_idx]
        return None

    def put(self, layer_idx: int, weights: LayerWeights):
        if len(self.cache) >= self.max_layers:
            # Evict LRU
            evicted_idx, evicted_weights = self.cache.popitem(last=False)
            del evicted_weights  # Free GPU memory
        self.cache[layer_idx] = weights
```

### 3. StreamingGPT: 스트리밍 모델

```python
class StreamingGPT:
    """
    레이어별 스트리밍 추론
    전체 모델을 GPU에 올리지 않음
    """

    def __init__(self, config: GPTConfig, weight_store: WeightStore):
        self.config = config
        self.store = weight_store
        self.cache = LayerCache(max_layers=4)  # 4 레이어만 캐시

        # 공유 버퍼 (재사용으로 할당 최소화)
        self.embed_buffer = None
        self.hidden_buffer = None

    def forward_streaming(self, tokens: cp.ndarray) -> cp.ndarray:
        """스트리밍 추론: 레이어별로 가중치 로드"""

        # Embeddings (항상 상주)
        x = self._embedding(tokens)

        # Prefetch first layer
        prefetch_stream = cp.cuda.Stream()
        self._prefetch_layer(0, prefetch_stream)

        for i in range(self.config.n_layer):
            # 현재 레이어 가중치 가져오기 (캐시 또는 로드)
            weights = self._get_layer_weights(i)

            # 다음 레이어 비동기 프리페치
            if i + 1 < self.config.n_layer:
                self._prefetch_layer(i + 1, prefetch_stream)

            # Tile 커널로 계산
            x = self._transformer_block(x, weights)

            # 캐시 정책에 따라 자동 관리
            # (오래된 레이어는 자동 evict)

        return self._lm_head(x)

    def _get_layer_weights(self, layer_idx: int) -> LayerWeights:
        """캐시 히트 또는 로드"""
        cached = self.cache.get(layer_idx)
        if cached is not None:
            return cached

        # Cache miss: 로드 필요
        weights = self.store.load_layer(layer_idx)
        self.cache.put(layer_idx, weights)
        return weights
```

---

## 메모리 비교

```
시나리오: GPT-2 XL (48 레이어, ~6GB)

┌─────────────────────────────────────────────────────────────┐
│ 기존 방식 (전체 로드)                                         │
│                                                              │
│ GPU Memory: [Layer 0][Layer 1]...[Layer 47][KV Cache][Act]  │
│             └──────────── 6GB ──────────────┘ + 2GB = 8GB   │
│                                                              │
│ 필요 GPU: 최소 10GB                                          │
│ Cold Start: ~10초 (전체 로드)                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ StreamTile 방식 (4 레이어 캐시)                               │
│                                                              │
│ GPU Memory: [Layer i][Layer i+1][Layer i+2][Pool][Act]      │
│             └───────── 500MB ─────────────┘ + 500MB = 1GB   │
│                                                              │
│ 필요 GPU: 2-4GB (6x 감소!)                                   │
│ Cold Start: ~1초 (임베딩만 로드)                             │
│ 추가 Latency: ~5% (프리페치로 오버랩)                        │
└─────────────────────────────────────────────────────────────┘
```

### 메모리 절감 효과

| 모델 | 기존 방식 | StreamTile (4-layer cache) | 절감 |
|------|----------|---------------------------|------|
| GPT-2 (117M) | 500MB | ~170MB | 3x |
| GPT-2 XL (1.5B) | 6GB | ~1GB | 6x |
| LLaMA-7B | 28GB | ~4GB | 7x |

---

## vLLM 영감: 추가 최적화

### 1. PagedWeights (PagedAttention 영감)

```python
class PagedWeightManager:
    """
    vLLM의 PagedAttention처럼 가중치를 페이지 단위로 관리
    """

    PAGE_SIZE = 4 * 1024 * 1024  # 4MB pages

    def __init__(self, gpu_memory_budget: int):
        self.budget = gpu_memory_budget
        self.page_table = {}  # virtual → physical
        self.free_pages = []

    def allocate_weight_pages(self, weight_size: int) -> List[int]:
        """가중치에 페이지 할당"""
        num_pages = (weight_size + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        return self._get_free_pages(num_pages)
```

### 2. Hot-Swap Layers (LoRA 영감)

```python
class HotSwappableGPT(StreamingGPT):
    """
    특정 레이어만 동적으로 교체 가능
    Fine-tuning 없이 행동 변경
    """

    def swap_layer(self, layer_idx: int, new_weights: LayerWeights):
        """런타임에 레이어 교체"""
        self.cache.invalidate(layer_idx)
        self.store.update_layer(layer_idx, new_weights)

    def apply_adapter(self, layer_idx: int, adapter: LoRAAdapter):
        """LoRA 어댑터 적용 (추가 메모리 최소)"""
        base = self._get_layer_weights(layer_idx)
        return base + adapter.delta  # Low-rank addition
```

### 3. Continuous Batching 지원

```python
class StreamingBatchGPT:
    """
    vLLM처럼 동적 배치 관리
    스트리밍 가중치 + 연속 배칭
    """

    def add_request(self, tokens: cp.ndarray, request_id: str):
        """새 요청 추가 (진행 중에도)"""
        self.pending_requests.append((request_id, tokens))

    def step(self) -> Dict[str, cp.ndarray]:
        """한 스텝 처리 (여러 요청 동시)"""
        # 같은 레이어의 요청들을 배치로 처리
        # 가중치 한 번 로드 → 여러 요청 처리
```

---

## 기술적 타당성 분석

### 가능한 것들 ✅

| 기능 | 타당성 | 구현 난이도 |
|------|--------|-------------|
| 레이어별 로드/언로드 | ✅ 높음 | 중간 |
| LRU 캐시 | ✅ 높음 | 낮음 |
| 비동기 프리페치 | ✅ 높음 (CUDA Stream) | 중간 |
| mmap 스타일 로드 | ✅ 높음 (CuPy DLPack) | 중간 |
| Hot-swap 레이어 | ✅ 높음 | 낮음 |

### 도전적인 것들 ⚠️

| 기능 | 도전 요소 | 해결 방향 |
|------|-----------|-----------|
| 프리페치 타이밍 | 계산 시간 예측 필요 | 프로파일링 기반 |
| 페이지 관리 | 메모리 단편화 | Slab allocator |
| 병렬 요청 처리 | 동기화 복잡성 | 레이어 단위 락 |

---

## 구현 로드맵

### Phase 1: 기본 스트리밍 (2-3주)

- [ ] WeightStore 구현 (mmap + 레이어 인덱싱)
- [ ] LayerCache 구현 (LRU)
- [ ] StreamingGPT 기본 버전
- [ ] 벤치마크: 메모리 vs 성능 트레이드오프

### Phase 2: 비동기 최적화 (2-3주)

- [ ] CUDA Stream 프리페치
- [ ] TMA 활용 (Hopper+)
- [ ] 오버랩 측정 및 최적화
- [ ] 자동 프리페치 타이밍

### Phase 3: 고급 기능 (4주+)

- [ ] PagedWeights
- [ ] Hot-swap layers
- [ ] LoRA 어댑터 지원
- [ ] Continuous batching

---

## 예상 API

```python
from cutile_gpt import StreamingGPT, GPTConfig
from cutile_gpt.streaming import WeightStore, StreamConfig

# 가중치 저장소 생성 (디스크 기반)
store = WeightStore.from_huggingface('gpt2-xl', cache_dir='./weights')

# 스트리밍 설정
config = StreamConfig(
    max_cached_layers=4,      # GPU에 최대 4개 레이어 유지
    prefetch_layers=2,        # 2개 레이어 미리 로드
    memory_budget_gb=2.0,     # GPU 메모리 예산
)

# 스트리밍 모델 생성
model = StreamingGPT(
    GPTConfig.gpt2_xl(),
    weight_store=store,
    stream_config=config,
)

# 추론 (메모리 효율적)
tokens = cp.array([[15496, 11, 616, 1438, 318]], dtype=cp.int32)
output = model.generate(tokens, max_new_tokens=100)

# 메모리 사용량 확인
print(f"GPU Memory: {model.memory_usage_mb():.1f} MB")  # ~1000 MB (6GB 대신)
```

---

## 결론

StreamTile은 cutileGPT의 "역발상" 철학을 메모리 관리에 적용합니다:

| 기존 패러다임 | StreamTile 패러다임 |
|--------------|-------------------|
| 모델 압축 | 인프라 경량화 |
| 정적 로딩 | 동적 스트리밍 |
| 전체 모델 상주 | 필요한 부분만 로드 |
| GPU 메모리 제약 | GPU 메모리 활용 |

이 접근법은:
- **6x 메모리 절감** 가능
- **기존 모델 그대로 사용** (압축 불필요)
- **Tile Programming과 자연스럽게 통합**
- **Edge/Serverless 배포에 적합**

---

## References

- [vLLM: Easy, Fast, and Cheap LLM Serving](https://github.com/vllm-project/vllm)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [NVIDIA TMA Documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [CuPy Memory Management](https://docs.cupy.dev/en/stable/user_guide/memory.html)

---

*Document Version: 1.0*
*Last Updated: 2026-01-30*
*Author: cutileGPT Team*
