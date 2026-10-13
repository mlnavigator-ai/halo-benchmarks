"""Data schemas for halo-benchmarks.

Defines the canonical JSON schema for all benchmark results, ensuring
reproducibility and comparability across runs on this hardware.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    """Physical machine specs."""
    cpu: str = ""               # e.g. "AMD RYZEN AI MAX+ 395"
    cpu_cores: int = 0          # physical cores
    cpu_threads_per_core: int = 1
    cpu_max_mhz: float = 0.0
    gpu_name: str = ""          # e.g. "AMD Radeon 8060S Graphics (RADV STRIX_HALO)"
    gpu_type: str = ""          # INTEGRATED_GPU / DISCRETE / CPU
    driver_version: str = ""    # e.g. "26.2.4"
    vulkan_version: str = ""    # e.g. "1.4.354"
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    platform: str = ""          # OS / kernel
    benchmark_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["benchmark_date"] = self.benchmark_date or datetime.now(timezone.utc).isoformat()
        return d


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    """Static model info."""
    name: str                   # preset alias, e.g. "gpt-oss-20b-heretic"
    variant: str = ""           # e.g. "ARA v3 MXFP4_MOE (mradermacher)"
    quantization: str = ""      # Q4_K_M, MXFP4_MOE, IQ4_XS, etc.
    file_size_gb: float = 0.0
    total_params_b: float = 0.0
    active_params_b: float = 0.0
    architecture: str = ""      # "MoE", "Dense", "Hybrid Gated DeltaNet + attention"
    vocab_size: int = 0
    n_blocks: int = 0
    sliding_window: int = 0     # 0 = full attention
    max_context: int = 0
    kv_heads: int = 0
    attn_heads: int = 0
    key_len: int = 64           # default
    value_len: int = 64         # default
    expert_count: int = 0       # MoE only
    expert_used: int = 0        # MoE only
    reasoning_format: str = ""  # "harmony", "deepseek", "none"
    abliteration: dict[str, Any] = field(default_factory=dict)
    chat_template_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Throughput test result
# ---------------------------------------------------------------------------

@dataclass
class ThroughputPoint:
    """Single throughput measurement."""
    n_predict: int              # requested output tokens
    actual_generated: int = 0   # actually produced
    prompt_tokens: int = 0
    cached_tokens: int = 0
    decode_tok_s: float = 0.0
    prefill_tok_s: float = 0.0
    decode_ms: float = 0.0
    prefill_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Reasoning test result
# ---------------------------------------------------------------------------

@dataclass
class ReasoningResult:
    """Single reasoning / logic test."""
    name: str                   # e.g. "bat_and_ball", "math_17x23"
    correct: bool = False
    answer: str = ""
    expected: str = ""
    tok_per_second: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Context window test result
# ---------------------------------------------------------------------------

@dataclass
class ContextResult:
    """Single context-window / needle-in-haystack test."""
    prompt_tokens: int = 0
    items_stored: int = 0
    retrieval_results: dict[str, Any] = field(default_factory=dict)
    correct: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Concurrency test result
# ---------------------------------------------------------------------------

@dataclass
class ConcurrencyResult:
    """Concurrency / eviction measurements."""
    scenario: str               # e.g. "9b_plus_20b", "120b_plus_9b"
    models: list[str] = field(default_factory=list)
    solo_throughput: dict[str, float] = field(default_factory=dict)  # model -> tok/s
    concurrent_throughput: dict[str, float] = field(default_factory=dict)
    degradation_pct: dict[str, float] = field(default_factory=dict)
    eviction_rule: str = ""     # e.g. "LRU"
    max_concurrent: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRun:
    """Complete benchmark result for one model."""
    hardware: HardwareProfile
    model: ModelMetadata
    throughput_points: list[ThroughputPoint] = field(default_factory=list)
    reasoning_results: list[ReasoningResult] = field(default_factory=list)
    context_results: list[ContextResult] = field(default_factory=list)
    concurrency_results: list[ConcurrencyResult] = field(default_factory=list)
    memory_rss_mb: float = 0.0
    memory_vm_peak_mb: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def avg_decode_tok_s(self) -> float:
        if not self.throughput_points:
            return 0.0
        valid = [p.decode_tok_s for p in self.throughput_points if p.decode_tok_s > 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_prefill_tok_s(self) -> float:
        if not self.throughput_points:
            return 0.0
        valid = [p.prefill_tok_s for p in self.throughput_points if p.prefill_tok_s > 0]
        return sum(valid) / len(valid) if valid else 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "hardware": self.hardware.to_dict(),
            "model": self.model.to_dict(),
            "throughput": [p.to_dict() for p in self.throughput_points],
            "reasoning": [r.to_dict() for r in self.reasoning_results],
            "context_window": [c.to_dict() for c in self.context_results],
            "concurrency": [c.to_dict() for c in self.concurrency_results],
            "memory": {
                "rss_mb": round(self.memory_rss_mb, 1),
                "vm_peak_mb": round(self.memory_vm_peak_mb, 1),
            },
            "notes": self.notes,
        }
        # Add summary stats
        d["summary"] = {
            "avg_decode_tok_s": round(self.avg_decode_tok_s, 1),
            "avg_prefill_tok_s": round(self.avg_prefill_tok_s, 1),
            "active_params_b": self.model.active_params_b,
            "total_params_b": self.model.total_params_b,
        }
        if self.model.active_params_b > 0:
            d["summary"]["tok_s_per_active_billion"] = round(
                self.avg_decode_tok_s / self.model.active_params_b, 2
            )
        return d

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved benchmark to {path}")

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkRun:
        """Load from JSON and reconstruct dataclass objects.

        Supports two formats:
        1. New schema (BenchmarkRun.to_dict() output) — has "summary", "throughput" arrays
        2. Legacy flat format — has "performance", reasoning as string dict, no "model" key
        """
        with open(path) as f:
            data = json.load(f)

        # Detect legacy vs new format
        if "performance" in data and "throughput" not in data:
            return cls._load_legacy(data)

        hw_data = data["hardware"]
        model_data = data["model"]
        summary = data.get("summary", {})

        run = cls(
            hardware=HardwareProfile(**{k: v for k, v in hw_data.items() if k != "benchmark_date"}),
            model=ModelMetadata(**{k: v for k, v in model_data.items()}),
        )
        run.hardware.benchmark_date = hw_data.get("benchmark_date", "")

        for tp in data.get("throughput", []):
            run.throughput_points.append(ThroughputPoint(**tp))
        for rr in data.get("reasoning", []):
            run.reasoning_results.append(ReasoningResult(**rr))
        for cr in data.get("context_window", []):
            run.context_results.append(ContextResult(**cr))
        for cc in data.get("concurrency", []):
            run.concurrency_results.append(ConcurrencyResult(**cc))

        mem = data.get("memory", {})
        run.memory_rss_mb = mem.get("rss_mb", 0.0)
        run.memory_vm_peak_mb = mem.get("vm_peak_mb", 0.0)
        run.notes = data.get("notes", [])

        return run

    @classmethod
    def _load_legacy(cls, data: dict[str, Any]) -> BenchmarkRun:
        """Load legacy flat-format benchmark JSON."""
        hw_data = data.get("hardware", {})
        perf = data.get("performance", {})

        # Parse throughput from legacy format into ThroughputPoint objects
        tp_dict = perf.get("throughput_tok_per_second", {})
        throughput_points = []
        for label, tok_s in tp_dict.items():
            if isinstance(tok_s, (int, float)):
                try:
                    n_predict = int(label.split("_")[0])
                except (ValueError, IndexError):
                    n_predict = 100  # default fallback
                throughput_points.append(ThroughputPoint(
                    n_predict=n_predict,
                    actual_generated=n_predict,
                    decode_tok_s=float(tok_s),
                    prefill_tok_s=0.0,
                ))

        # Parse reasoning from legacy string dict into ReasoningResult objects
        reason_dict = data.get("reasoning", {})
        reasoning_results = []
        for name, result in reason_dict.items():
            if isinstance(result, str):
                is_correct = "CORRECT" in result.upper() or "GENERATED ITERATIVE" in result
                reasoning_results.append(ReasoningResult(
                    name=name,
                    correct=is_correct,
                    answer=result,
                    expected="—" if "CORRECT" not in result else result.split("(")[0].strip(),
                ))

        # Parse context window results
        ctx_data = data.get("context_window", {})
        context_results = []
        context_results.append(ContextResult(
            items_stored=5,
            retrieval_results={
                "theoretical_max_tokens": ctx_data.get("theoretical_max_tokens"),
                "effective_attention_window": ctx_data.get("effective_attention_window"),
                "retrieval_5_items": ctx_data.get("retrieval_5_items"),
                "note": ctx_data.get("note"),
            },
            correct="CORRECT" in str(ctx_data.get("retrieval_5_items", "")).upper(),
        ))

        # Hardware
        hw = HardwareProfile(
            gpu_name=hw_data.get("gpu", ""),
            gpu_type="INTEGRATED_GPU" if "integrated" in str(hw_data.get("gpu_type", "")).lower() else "",
            ram_total_gb=float(hw_data.get("ram_total_gb", 0)),
            ram_available_gb=0.0,
        )

        # Model metadata — legacy has these at top level
        model = ModelMetadata(
            name=data.get("model", ""),
            variant=data.get("variant", ""),
            architecture=data.get("architecture", ""),
            quantization="",
            abliteration=data.get("abliteration", {}),
        )

        mem = perf.get("memory_rss_mb", 0)
        vm_peak = perf.get("vm_peak_mb", 0)

        run = cls(
            hardware=hw,
            model=model,
            throughput_points=throughput_points,
            reasoning_results=reasoning_results,
            context_results=context_results,
            memory_rss_mb=float(mem),
            memory_vm_peak_mb=float(vm_peak),
            notes=[ctx_data.get("note", "")],
        )

        return run


# ---------------------------------------------------------------------------
# Cross-model comparison manifest
# ---------------------------------------------------------------------------

@dataclass
class ComparisonManifest:
    """Side-by-side summary across all models."""
    hardware: str = ""
    hardware_profile: dict[str, Any] = field(default_factory=dict)
    runs: list[BenchmarkRun] = field(default_factory=list)

    @property
    def comparison_table(self) -> dict[str, dict[str, Any]]:
        """Compact table keyed by model name."""
        table: dict[str, dict[str, Any]] = {}
        for run in self.runs:
            m = run.model
            s = {
                "total_params_b": m.total_params_b,
                "active_params_b": m.active_params_b,
                "quantization": m.quantization,
                "file_size_gb": m.file_size_gb,
                "architecture": m.architecture,
                "sliding_window": m.sliding_window,
                "max_context": m.max_context,
                "avg_decode_tok_s": round(run.avg_decode_tok_s, 1),
                "avg_prefill_tok_s": round(run.avg_prefill_tok_s, 1),
                "rss_mb": round(run.memory_rss_mb, 0),
                "vm_peak_mb": round(run.memory_vm_peak_mb, 0),
            }
            if m.active_params_b > 0:
                s["tok_s_per_active_billion"] = round(
                    run.avg_decode_tok_s / m.active_params_b, 2
                )
            table[m.name] = s
        return table

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "hardware": self.hardware,
            "hardware_profile": self.hardware_profile,
            "comparison_table": self.comparison_table,
            "details": [r.to_dict() for r in self.runs],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_dir(cls, raw_dir: str | Path) -> ComparisonManifest:
        """Load all .json benchmark files from a directory."""
        manifest = cls()
        raw_path = Path(raw_dir)
        for json_file in sorted(raw_path.glob("*.json")):
            try:
                run = BenchmarkRun.load(json_file)
                # Fallback: if model name is empty, use filename stem
                if not run.model.name:
                    run.model.name = json_file.stem.replace("_", "-")
                manifest.runs.append(run)
            except Exception as e:
                print(f"Skipping {json_file}: {e}")
        if manifest.runs:
            hw = manifest.runs[0].hardware
            manifest.hardware = f"{hw.cpu} / {hw.gpu_name}"
            manifest.hardware_profile = hw.to_dict()
        return manifest
