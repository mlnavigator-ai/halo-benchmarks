"""Migrate legacy benchmark JSON files to the canonical BenchmarkRun schema.

Each old file has its own ad-hoc structure. This script reads them all and
writes a normalized version into data/normalized/ that the comparison tools
can consume reliably.

Usage:
    python -m benchmarks.migrate data/raw/ --output data/normalized/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.schema import (
    BenchmarkRun,
    ConcurrencyResult,
    ContextResult,
    HardwareProfile,
    ModelMetadata,
    ReasoningResult,
    ThroughputPoint,
)


def _parse_memory_mb(text: str | float) -> float:
    """Extract MB value from string like '~940 MB' or '16172'."""
    if isinstance(text, (int, float)):
        return float(text)
    m = re.search(r'([\d.]+)', str(text))
    return float(m.group(1)) if m else 0.0


def _parse_params_b(text: str) -> float:
    """Extract params in billions from string like '~8.8B' or '21B'."""
    m = re.search(r'([\d.]+)\s*[Bb]', text)
    return float(m.group(1)) if m else 0.0


def _detect_architecture(text: str) -> tuple[str, float, float]:
    """Detect architecture type and params from description string."""
    text = text.lower()
    if "moe" in text:
        # Try to extract active/total params
        total_m = re.search(r'total[^B]*([\d.]+)\s*B', text)
        active_m = re.search(r'active[^B]*([\d.]+)\s*B', text)
        return "MoE", _parse_params_b(active_m.group(1)) if active_m else 0, _parse_params_b(total_m.group(1)) if total_m else 0
    elif "delta" in text or "gated" in text:
        total_m = re.search(r'([\d.]+)\s*B', text)
        return "Gated DeltaNet", _parse_params_b(total_m.group(1)) if total_m else 0, 0
    elif "ssm" in text or "hybrid" in text:
        total_m = re.search(r'total[^B]*([\d.]+)\s*B', text)
        return "Hybrid dense/SSM", _parse_params_b(total_m.group(1)) if total_m else 0, 0
    else:
        total_m = re.search(r'([\d.]+)\s*B', text)
        return "Dense", _parse_params_b(total_m.group(1)) if total_m else 0, 0


# ---------------------------------------------------------------------------
# Legacy format handlers
# ---------------------------------------------------------------------------

def migrate_qwen35(data: dict, filename: str) -> BenchmarkRun:
    """Migrate qwen3.5-9b heretic format (structured nested dicts)."""
    arch = data.get("architecture", {})
    total_params = _parse_params_b(arch.get("total_params", "0B"))

    # Throughput details
    throughput_details = []
    for d in data.get("throughput", {}).get("details", []):
        throughput_details.append(ThroughputPoint(
            n_predict=d.get("predict", 100),
            actual_generated=d.get("predict", 100),
            decode_tok_s=d.get("tok_s", 0),
            prefill_tok_s=0,
        ))

    # Reasoning tests
    reasoning = []
    for test_name, result in data.get("reasoning_tests", {}).items():
        if isinstance(result, bool):
            reasoning.append(ReasoningResult(name=test_name, correct=result))
        elif isinstance(result, dict):
            reasoning.append(ReasoningResult(
                name=test_name,
                correct=result.get("correct", result.get("pass", False)),
                answer=str(result.get("answer", "")),
            ))
        elif isinstance(result, str):
            reasoning.append(ReasoningResult(name=test_name, correct="CORRECT" in result.upper(), answer=result))

    # Context window
    context = []
    for test_name, result in data.get("context_window", {}).items():
        if isinstance(result, dict):
            # Parse items_stored from names like "#7_of_10" or "#25_of_50"
            m = re.search(r'#\d+_\w+(\d+)', test_name)
            items = int(m.group(1)) if m else 10
            context.append(ContextResult(
                items_stored=items,
                retrieval_results={"answer": str(result.get("answer", "")), "correct": result.get("correct", False)},
                correct=result.get("correct", False),
            ))

    # Hardware - use default Strix Halo profile
    hw = HardwareProfile(
        cpu="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
        gpu_name="AMD Radeon 8060S Graphics (RADV STRIX_HALO)",
        gpu_type="INTEGRATED_GPU",
        ram_total_gb=122.8,
        benchmark_date=data.get("benchmark_date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
    )

    model = ModelMetadata(
        name=data.get("model", Path(filename).stem),
        quantization=data.get("quant", ""),
        total_params_b=total_params,
        active_params_b=total_params,  # qwen3.5 is dense
        architecture=arch.get("type", ""),
        max_context=arch.get("max_context", 0),
    )

    rss = _parse_memory_mb(data.get("memory", {}).get("rss_peak_mb", 0))

    return BenchmarkRun(
        hardware=hw, model=model,
        throughput_points=throughput_details,
        reasoning_results=reasoning,
        context_results=context,
        memory_rss_mb=rss,
        notes=[f"Architecture: {arch.get('type', '')}"],
    )


def migrate_gptoss_20b(data: dict, filename: str | Path) -> BenchmarkRun:
    """Migrate gpt-oss-20b heretic format (performance dict with tok_per_second)."""
    arch_data = data.get("architecture", "")
    if isinstance(arch_data, dict):
        # Structured architecture data — extract directly
        arch_type = arch_data.get("type", "MoE")
        active_b = _parse_params_b(str(arch_data.get("active_params", "0B")))
        total_b = _parse_params_b(str(arch_data.get("total_params", "0B")))
    else:
        arch_text = str(arch_data) if not isinstance(arch_data, dict) else json.dumps(arch_data)
        arch_type, active_b, total_b = _detect_architecture(arch_text)

    # Parse throughput
    tp_dict = data.get("performance", {}).get("throughput_tok_per_second", {})
    throughput = []
    for label, tok_s in tp_dict.items():
        if isinstance(tok_s, (int, float)):
            try:
                n = int(str(label).split("_")[0])
            except (ValueError, IndexError):
                n = 100
            throughput.append(ThroughputPoint(n_predict=n, actual_generated=n, decode_tok_s=float(tok_s)))

    # Reasoning
    reasoning = []
    for name, result in data.get("reasoning", {}).items():
        if isinstance(result, str):
            correct = "CORRECT" in result.upper() or "GENERATED ITERATIVE" in result
            reasoning.append(ReasoningResult(name=name, correct=correct, answer=result))

    # Context
    ctx = data.get("context_window", {})
    context = [ContextResult(
        items_stored=5,
        retrieval_results={k: v for k, v in ctx.items() if k != "retrieval_5_items"},
        correct="CORRECT" in str(ctx.get("retrieval_5_items", "")).upper(),
    )]

    hw_data = data.get("hardware", {})
    gpu_type_val = str(hw_data.get("gpu_type", ""))
    hw = HardwareProfile(
        gpu_name=hw_data.get("gpu", ""),
        gpu_type="INTEGRATED_GPU" if "integrated" in gpu_type_val.lower() else "",
        ram_total_gb=float(hw_data.get("ram_total_gb", 0)),
    )

    model = ModelMetadata(
        name=data.get("model", str(filename).rsplit(".", 1)[0]),
        variant=data.get("variant", ""),
        quantization="",
        total_params_b=total_b,
        active_params_b=active_b,
        architecture=arch_type,
        abliteration=data.get("abliteration", {}),
    )

    perf = data.get("performance", {})
    return BenchmarkRun(
        hardware=hw, model=model,
        throughput_points=throughput, reasoning_results=reasoning, context_results=context,
        memory_rss_mb=float(perf.get("memory_rss_mb", 0)),
        memory_vm_peak_mb=float(perf.get("vm_peak_mb", 0)),
    )


def migrate_gptoss_120b(data: dict, filename: str) -> BenchmarkRun:
    """Migrate gpt-oss-120b format (context_window is int, throughput nested differently)."""
    arch_text = data.get("architecture", "")
    arch_type, active_b, total_b = _detect_architecture(arch_text)

    # Throughput - decode values
    tp_dict = data.get("throughput", {}).get("decode_tok_s", {})
    throughput = []
    for label, tok_s in tp_dict.items():
        if isinstance(tok_s, (int, float)):
            # Extract output length from label like "short_output_16" or "medium_output_256"
            m = re.search(r'(\d+)', label)
            n = int(m.group(1)) if m else 100
            throughput.append(ThroughputPoint(n_predict=n, actual_generated=n, decode_tok_s=float(tok_s)))

    # Reasoning
    reasoning = []
    for name, result in data.get("reasoning_quality", {}).items():
        correct = "Correct" in str(result) or "good" in str(result).lower()
        reasoning.append(ReasoningResult(name=name, correct=correct, answer=str(result)))

    # Context - context_window is just an int (max tokens)
    ctx_max = data.get("context_window", 0)
    context = [ContextResult(
        items_stored=1,
        retrieval_results={"max_tokens": ctx_max, "note": "SWA limits effective attention"},
    )]

    hw = HardwareProfile(
        cpu="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
        gpu_name="AMD Radeon 8060S Graphics (RADV STRIX_HALO)",
        gpu_type="INTEGRATED_GPU",
        ram_total_gb=122.8,
    )

    mem = data.get("memory_footprint", {})
    model = ModelMetadata(
        name=data.get("model", Path(filename).stem),
        variant=data.get("variant", ""),
        total_params_b=total_b or 117,
        active_params_b=active_b or 5.1,
        architecture=arch_type,
    )

    return BenchmarkRun(
        hardware=hw, model=model,
        throughput_points=throughput, reasoning_results=reasoning, context_results=context,
        memory_rss_mb=_parse_memory_mb(mem.get("estimated_runtime_gb", "0")) * 1024 if mem.get("estimated_runtime_gb") else 0,
    )


def migrate_qwen38(data: dict, filename: str | Path) -> BenchmarkRun:
    """Migrate qwen3.8 uncensored format (structured JSON with throughput strings)."""
    hw = HardwareProfile(
        gpu_name="AMD Radeon 8060S Graphics (RADV STRIX_HALO)",
        gpu_type="INTEGRATED_GPU",
        ram_total_gb=122.8,
    )

    arch_text = data.get("architecture", "")
    if isinstance(arch_text, dict):
        arch_text = json.dumps(arch_text)
    total_b = _parse_params_b(str(arch_text))

    model = ModelMetadata(
        name=data.get("model", str(filename).rsplit(".", 1)[0]),
        quantization=data.get("quant", ""),
        architecture="Gated DeltaNet",
        total_params_b=total_b,
    )

    # Parse throughput — values are strings like "12.1 tok/s"
    tp_dict = data.get("throughput", {})
    throughput = []
    for label, val in tp_dict.items():
        if isinstance(val, str):
            m = re.search(r'([\d.]+)', str(val))
            tok_s = float(m.group(1)) if m else 0
        elif isinstance(val, (int, float)):
            tok_s = float(val)
        else:
            continue
        try:
            n = int(str(label).split("_")[0])
        except (ValueError, IndexError):
            n = 50
        throughput.append(ThroughputPoint(n_predict=n, actual_generated=n, decode_tok_s=tok_s))

    # Reasoning
    reasoning = []
    for name, result in data.get("reasoning", {}).items():
        if isinstance(result, str):
            correct = "CORRECT" in result.upper() and "WRONG" not in result.upper()
            reasoning.append(ReasoningResult(name=name, correct=correct, answer=result))

    # Context window test
    ctx = data.get("context_window_test", {})
    context = []
    if ctx:
        items = ctx.get("items_stored", 50)
        context.append(ContextResult(
            items_stored=items,
            retrieval_results={k: str(v) for k, v in ctx.items() if k not in ("prompt_tokens", "items_stored")},
            correct=False,  # qwen3.8 failed most retrievals at this scale
        ))

    return BenchmarkRun(
        hardware=hw, model=model,
        throughput_points=throughput, reasoning_results=reasoning, context_results=context,
        memory_rss_mb=_parse_memory_mb(data.get("memory_rss", "0")),
    )


def migrate_concurrency(data: dict, filename: str) -> BenchmarkRun:
    """Migrate concurrency test results into a special BenchmarkRun."""
    hw = HardwareProfile(
        gpu_name="AMD Radeon 8060S Graphics (RADV STRIX_HALO)",
        gpu_type="INTEGRATED_GPU",
        ram_total_gb=122.8,
    )

    model = ModelMetadata(name="concurrency_test")

    concurrency_results = []
    for item in data.get("results", [data]):
        if isinstance(item, dict):
            concurrency_results.append(ConcurrencyResult(
                scenario=item.get("scenario", item.get("models", "unknown")),
                models=item.get("models", []),
                solo_throughput=item.get("solo", item.get("solo_throughput", {})),
                concurrent_throughput=item.get("concurrent", item.get("concurrent_throughput", {})),
                degradation_pct=item.get("degradation", item.get("degradation_pct", {})),
            ))

    return BenchmarkRun(
        hardware=hw, model=model,
        concurrency_results=concurrency_results,
    )


# Format handler registry - keyed by filename patterns
FORMAT_HANDLERS = [
    (lambda f: "qwen3.5" in f.lower() and "heretic" in f.lower(), migrate_qwen35),
    (lambda f: "gpt_oss_120b" in f.lower() or "gptoss120b" in f.lower(), migrate_gptoss_120b),
    (lambda f: "gpt_oss_20b" in f.lower() or "gptoss20b" in f.lower(), migrate_gptoss_20b),
    (lambda f: "qwen3.8" in f.lower() and "uncensored" in f.lower(), migrate_qwen38),
    (lambda f: "concurrency" in f.lower(), migrate_concurrency),
]


def detect_handler(filename: str):
    """Find the migration handler for a given filename."""
    basename = Path(filename).stem.lower()
    for predicate, handler in FORMAT_HANDLERS:
        if predicate(basename):
            return handler
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def migrate_all(raw_dir: str | Path, output_dir: str | Path) -> list[BenchmarkRun]:
    """Migrate all legacy files to normalized format."""
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    migrated = []
    for json_file in sorted(raw_path.glob("*.json")):
        handler = detect_handler(json_file.name)
        if not handler:
            print(f"  SKIP {json_file.name} (no handler)")
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)
            run = handler(data, json_file.name)
            out_file = out_path / f"{run.model.name.replace(' ', '_')}.json"
            run.save(out_file)
            migrated.append(run)
            print(f"  OK   {json_file.name} -> {out_file.name}")
        except Exception as e:
            print(f"  FAIL {json_file.name}: {e}")

    return migrated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy benchmark files")
    parser.add_argument("raw_dir", help="Directory with legacy JSON files")
    parser.add_argument("--output", "-o", default="data/normalized", help="Output directory")

    args = parser.parse_args(argv)

    print(f"\nMigrating {args.raw_dir} -> {args.output}")
    migrated = migrate_all(args.raw_dir, args.output)
    print(f"\nMigrated {len(migrated)} files:")
    for run in migrated:
        print(f"  - {run.model.name}: {run.avg_decode_tok_s:.1f} tok/s, "
              f"{len(run.reasoning_results)} reasoning tests")


if __name__ == "__main__":
    main()
