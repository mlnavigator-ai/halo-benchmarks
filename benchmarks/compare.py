"""Cross-model comparison for halo-benchmarks.

Usage:
    python -m benchmarks.compare data/raw/
    python -m benchmarks.compare data/raw/ --output data/processed/comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.schema import BenchmarkRun, ComparisonManifest


def format_table(table: dict[str, dict[str, Any]]) -> str:
    """Pretty-print a comparison table as aligned text."""
    if not table:
        return "No models to compare."

    # Collect all keys and model names
    models = sorted(table.keys())
    all_keys = []
    for m in models:
        for k in table[m]:
            if k not in all_keys:
                all_keys.append(k)

    # Calculate column widths
    col_widths = {}
    for key in all_keys:
        col_widths[key] = max(
            len(str(key)),
            *[len(str(table[m].get(key, ""))) for m in models],
        )

    # Build table
    lines = []
    header = "Model".ljust(28)
    for key in all_keys:
        header += " | ".join([key.rjust(col_widths[key])])
    lines.append(header)
    lines.append("-" * len(header))

    for model in models:
        row = model.ljust(28)
        for key in all_keys:
            val = table[model].get(key, "")
            formatted = _format_value(val)
            row += " | ".join([str(formatted).rjust(col_widths[key])])
        lines.append(row)

    return "\n".join(lines)


def _format_value(val: Any) -> str:
    """Format a value for display."""
    if isinstance(val, float):
        if abs(val) < 0.01:
            return "0"
        if abs(val) < 100:
            return f"{val:.2f}"
        return f"{val:.0f}"
    if isinstance(val, bool):
        return "YES" if val else "NO"
    if isinstance(val, list):
        if not val:
            return "-"
        return ", ".join(str(v) for v in val[:3])
    if isinstance(val, dict):
        if not val:
            return "-"
        items = [f"{k}: {v}" for k, v in val.items()]
        return "; ".join(items[:3])
    if val is None:
        return "-"
    return str(val)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="halo-benchmarks comparison")
    parser.add_argument("raw_dir", help="Directory containing raw benchmark JSON files")
    parser.add_argument("--output", "-o", help="Output comparison JSON path")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of table")

    args = parser.parse_args(argv)

    raw_path = Path(args.raw_dir)
    if not raw_path.exists():
        print(f"Error: {raw_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    manifest = ComparisonManifest.from_dir(raw_path)

    if args.json or args.output:
        manifest.save(args.output or str(raw_path.parent / "processed" / "comparison.json"))
        print(f"Saved comparison to {args.output or raw_path.parent / 'processed' / 'comparison.json'}")

    # Always print table to stdout
    table = manifest.comparison_table
    print("\n=== HALO-BENCHMARKS COMPARISON TABLE ===\n")
    print(format_table(table))

    # Print reasoning summary
    print("\n\n=== REASONING RESULTS ===\n")
    for run in manifest.runs:
        name = run.model.name
        correct = sum(1 for r in run.reasoning_results if r.correct)
        total = len(run.reasoning_results)
        pct = f"{correct}/{total}" if total else "0/0"
        print(f"  {name}: {pct} reasoning tests passed")
        for r in run.reasoning_results:
            status = "PASS" if r.correct else "FAIL"
            print(f"    [{status}] {r.name}: expected={r.expected}, got={r.answer}")

    # Print throughput summary
    print("\n\n=== THROUGHPUT SUMMARY ===\n")
    for run in manifest.runs:
        name = run.model.name
        avg_decode = run.avg_decode_tok_s
        avg_prefill = run.avg_prefill_tok_s
        active_b = run.model.active_params_b or run.model.total_params_b
        efficiency = avg_decode / active_b if active_b > 0 else 0
        print(f"  {name}:")
        print(f"    avg decode:     {avg_decode:.1f} tok/s")
        print(f"    avg prefill:    {avg_prefill:.1f} tok/s")
        print(f"    active params:  {active_b:.1f}B")
        print(f"    efficiency:     {efficiency:.2f} tok/s per active billion params")


if __name__ == "__main__":
    main()
