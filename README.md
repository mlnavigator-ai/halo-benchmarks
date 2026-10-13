# Halo Benchmarks

LLM inference benchmarks on **AMD Strix Halo** integrated GPU hardware (Radeon 8060S, UMA).

## Hardware

| Component | Specification |
|---|---|
| **CPU** | AMD RYZEN AI MAX+ 395 (16C/32T, up to 5.2 GHz) |
| **GPU** | AMD Radeon 8060S Graphics (RADV STRIX_HALO, Vulkan 1.4.354) |
| **Driver** | Mesa 26.2.4 - kisak-mesa PPA (radv) |
| **RAM** | 122.8 GB UMA (shared CPU/GPU) |
| **Swap** | 8 GB |
| **Platform** | Linux 7.0.0-38-generic |

## Models Benchmarked

| Model | Architecture | Params (Total/Active) | Quant | File Size | Decode (tok/s) | Efficiency* |
|---|---|---|---|---|---|---|
| **gpt-oss-20b-heretic** | MoE | 21B / 3.6B | MXFP4_MOE | ~12 GB | **56.6** | **15.72 tok/s per active B** |
| **gpt-oss-120b** | MoE | 117B / 5.1B | — | ~61 GB | **46.3** | **9.08 tok/s per active B** |
| **qwen3.5-9b-heretic** | Hybrid dense/SSM | 8.8B / 8.8B | Q4_K_M | 5.2 GB | **34.5** | 3.92 |
| **qwen3.8-27b-uncensored** | Gated DeltaNet | 27B / 27B | IQ4_XS | 15 GB | **12.0** | 0.45 |

*\*Efficiency = avg decode tok/s ÷ active parameters in billions*

## Key Findings

### MoE dominates on UMA hardware
The gpt-oss-20b (3.6B active) achieves **15.72 tok/s per active billion params** — 4x the qwen3.8 and 2.7x the qwen3.5. On integrated GPU with shared memory, MoE's sparse activation pattern minimizes data movement across the memory bus, making it ideal for this form factor.

### gpt-oss-120b fits — barely
At ~67 GB runtime (61 GB weights + KV cache headroom), the 117B model squeezes into 122 GB UMA with ~55 GB left for OS and a second model. It still delivers 46 tok/s decode and passes all reasoning tests.

### Sliding Window Attention limits context
Both gpt-oss models use 128-token SWA — they can accept massive prompts but only attend to the last 128 tokens. Needle-in-haystack retrieval works for short lists (5 items) but degrades beyond that. The qwen3.5 has no SWA and supports 262K context.

### Concurrency: ~30-40% throughput degradation
Two models running simultaneously on the same integrated GPU show 28-39% throughput loss each. The LLAMA_ARG_MODELS_MAX=2 router confirms LRU eviction — only 2 models can be fully loaded at once.

### Reasoning quality across architectures
| Model | Bat/Ball | Math (17×23+42) | Fibonacci Code | Sheep Riddle |
|---|---|---|---|---|
| gpt-oss-20b-heretic | ✅ $0.05 | ✅ 433 | ✅ Iterative | — |
| gpt-oss-120b | ✅ $0.05 | ✅ 391 (showed work) | ✅ Binary search | ✅ Step-by-step |
| qwen3.5-9b-heretic | ✅ $0.05 | ✅ 433 | ✅ Full implementation | ✅ 9 |
| qwen3.8-27b-uncensored | ⚠️ Incomplete | ❌ 393 | ⚠️ Base cases only | — |

## Repository Structure

```
halo-benchmarks/
├── benchmarks/              # Python package
│   ├── __init__.py
│   ├── schema.py            # Canonical dataclasses (BenchmarkRun, etc.)
│   ├── runner.py            # Live benchmark runner (hits llama-server API)
│   ├── compare.py           # Cross-model comparison CLI
│   └── migrate.py           # Legacy format → canonical schema migration
├── data/
│   ├── raw/                 # Original benchmark JSON files (various formats)
│   ├── normalized/          # Migrated to canonical BenchmarkRun schema
│   └── processed/           # Aggregated outputs (comparison.json, etc.)
├── scripts/                 # Ad-hoc analysis scripts
├── docs/                    # Documentation
├── pyproject.toml           # Package config + CLI entry points
└── README.md                # This file
```

## Usage

### Compare all models in a directory
```bash
python3 -m benchmarks.compare data/normalized/
```

### Run live benchmark against llama-server
```bash
python3 -m benchmarks.runner --preset gpt-oss-20b-heretic \
    --server http://127.0.0.1:1244 \
    --output data/raw/gpt_oss_20b_live.json
```

### Migrate legacy benchmark files to canonical schema
```bash
python3 -m benchmarks.migrate data/raw/ --output data/normalized/
```

### CLI entry points (after `pip install -e .`)
```bash
halo-compare data/normalized/        # Cross-model comparison table
halo-bench --preset gpt-oss-20b      # Run live benchmark
halo-analyze                         # Analysis tools (future)
```

## Methodology

### Throughput
Measures decode speed at output lengths 16, 64, 128, 256, and 512 tokens. Reports average across all lengths. Prefill speed measured separately from prompt processing.

### Reasoning Tests
1. **Bat & Ball** — Classic cognitive reflection test ($1.10 total, bat costs $1 more than ball)
2. **Math (17×23+42)** — Multi-step arithmetic requiring order of operations
3. **Fibonacci Code** — Iterative implementation with docstring and type hints
4. **Sheep Riddle** — Lateral thinking ("all but 9 run away")

### Context Window
Needle-in-haystack retrieval: list numbered items, ask for specific position. Tests at 5, 10, and 50 item scales.

### Concurrency
Solo baselines → concurrent requests (parallel threads) → degradation calculation. Eviction tested by loading a 3rd model with LLAMA_ARG_MODELS_MAX=2.

## License

MIT
