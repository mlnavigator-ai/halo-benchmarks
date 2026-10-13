"""Benchmark runner for halo-benchmarks.

Queries a running llama.cpp router (llama-server) and collects:
- Throughput at multiple output lengths
- Reasoning tests (bat/ball, math, fibonacci, riddle)
- Context window / needle-in-haystack retrieval
- Concurrency / eviction behavior

Usage:
    python -m benchmarks.runner --preset gpt-oss-20b-heretic --output data/raw/gpt_oss_20b.json
    python -m benchmarks.runner --preset qwen3.5-9b-heretic --server http://127.0.0.1:1244
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests  # pip install requests; falls back to urllib if missing

try:
    import requests  # noqa: F401
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from benchmarks.schema import (
    BenchmarkRun,
    ComparisonManifest,
    ConcurrencyResult,
    ContextResult,
    HardwareProfile,
    ModelMetadata,
    ReasoningResult,
    ThroughputPoint,
)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(server: str, path: str, payload: dict, timeout: int = 120) -> dict | None:
    """POST JSON and return parsed response, or None on failure."""
    url = f"{server}{path}"
    if HAS_REQUESTS:
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  POST {url} failed: {e}")
            return None
    else:
        # Fallback to curl
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            fpath = f.name
        try:
            cmd = ["curl", "-s", "-X", "POST", url,
                   "-H", "Content-Type: application/json",
                   "-d", f"@{fpath}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                print(f"  curl {url} failed: {result.stderr[:200]}")
                return None
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                print(f"  curl returned non-JSON: {result.stdout[:200]}")
                return None
        finally:
            Path(fpath).unlink(missing_ok=True)


def _get(server: str, path: str, timeout: int = 30) -> dict | None:
    """GET JSON and return parsed response."""
    url = f"{server}{path}"
    if HAS_REQUESTS:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  GET {url} failed: {e}")
            return None
    else:
        cmd = ["curl", "-s", url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return None


def _wait_for_loaded(server: str, preset: str, timeout: int = 300, step: int = 5) -> bool:
    """Block until a preset is loaded or timeout."""
    elapsed = 0
    while elapsed < timeout:
        data = _get(server, "/models")
        if data:
            for m in data.get("data", []):
                if preset.replace("-", "") in m["id"].replace("-", "") or preset in m["id"]:
                    if m["status"]["value"] == "loaded":
                        return True
                    elif m["status"]["value"] == "loading":
                        time.sleep(step)
                        elapsed += step
        time.sleep(step)
        elapsed += step
    print(f"  Timeout waiting for {preset} to load ({timeout}s)")
    return False


def _get_port(server: str, preset: str) -> str | None:
    """Get the port for a loaded preset."""
    data = _get(server, "/models")
    if not data:
        return None
    for m in data.get("data", []):
        if preset.replace("-", "") in m["id"].replace("-", "") or preset in m["id"]:
            if m["status"]["value"] == "loaded":
                args = m["status"].get("args", [])
                try:
                    idx = args.index("--port")
                    return args[idx + 1]
                except (ValueError, IndexError):
                    pass
    return None


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_hardware() -> HardwareProfile:
    """Detect and populate hardware profile from the system."""
    hw = HardwareProfile(benchmark_date="")

    # CPU
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        for line in cpuinfo.split("\n"):
            if line.startswith("model name"):
                hw.cpu = line.split(":", 1)[1].strip()
            elif line.startswith("cpu(s)"):
                hw.cpu_cores = int(line.split(":")[1].strip())
    except Exception:
        pass

    # lscpu for core/thread counts
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10)
        for line in result.stdout.split("\n"):
            if "Core(s) per socket" in line:
                hw.cpu_cores = int(line.split(":")[1].strip())
            elif "Thread(s) per core" in line:
                hw.cpu_threads_per_core = int(line.split(":")[1].strip())
            elif "CPU max MHz" in line:
                hw.cpu_max_mhz = float(line.split(":")[1].strip().split(".")[0])
    except Exception:
        pass

    # GPU from vulkaninfo
    try:
        result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10)
        lines = result.stdout.split("\n")
        in_gpu0 = False
        for line in lines:
            if "GPU0:" in line:
                in_gpu0 = True
            elif in_gpu0 and line.strip().startswith("GPU1"):
                break
            if in_gpu0:
                if "deviceName" in line:
                    hw.gpu_name = line.split("=")[1].strip()
                elif "driverVersion" in line:
                    hw.driver_version = line.split("=")[1].strip()
                elif "apiVersion" in line and not hw.vulkan_version:
                    hw.vulkan_version = line.split("=")[1].strip()
                elif "deviceType" in line:
                    dt = line.split("=")[1].strip()
                    hw.gpu_type = "INTEGRATED_GPU" if "INTEGRATED" in dt else "OTHER"
    except Exception:
        pass

    # RAM
    try:
        result = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=10)
        for line in result.stdout.split("\n"):
            if line.startswith("Mem:"):
                parts = line.split()
                hw.ram_total_gb = float(parts[1]) / 1024
                hw.ram_available_gb = float(parts[3]) / 1024
            elif line.startswith("Swap:"):
                parts = line.split()
                hw.swap_total_gb = float(parts[1]) / 1024
                hw.swap_used_gb = float(parts[2]) / 1024
    except Exception:
        pass

    # Platform
    try:
        result = subprocess.run(["uname", "-r"], capture_output=True, text=True, timeout=5)
        hw.platform = f"Linux {result.stdout.strip()}"
    except Exception:
        hw.platform = "Linux"

    return hw


# ---------------------------------------------------------------------------
# Benchmark phases
# ---------------------------------------------------------------------------

def benchmark_throughput(server: str, port: str, model_name: str) -> list[ThroughputPoint]:
    """Measure decode throughput at multiple output lengths."""
    print(f"  [{model_name}] Throughput...")
    prompts = [
        "Write code for a function that computes the factorial of a number. Include docstring and type hints.",
        "Explain the difference between HTTP/2 and HTTP/3 in detail, covering multiplexing, header compression, and security.",
        "Write a Python class for a thread-safe LRU cache with configurable max size, get/set operations, and clear method.",
    ]

    points = []
    for n_predict in [16, 64, 128, 256, 512]:
        prompt = prompts[n_predict % len(prompts)]
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.7,
            "stream": False,
        }
        resp = _post(server, f"/completion", payload, timeout=120)
        if not resp:
            continue

        toks_out = resp.get("tokens_extracted", 0)
        toks_prompt = resp.get("tokens_prompt", 0)
        cached = resp.get("cached_kv_matches", 0)
        timings = resp.get("timings", {})

        point = ThroughputPoint(
            n_predict=n_predict,
            actual_generated=toks_out,
            prompt_tokens=toks_prompt,
            cached_tokens=cached,
            decode_tok_s=timings.get("predicted_per_second", 0) or timings.get("decode", 0),
            prefill_tok_s=timings.get("prompt_per_second", 0) or timings.get("load", 0),
            decode_ms=timings.get("predicted_ms", 0) or timings.get("decode_ms", 0),
            prefill_ms=timings.get("prompt_ms", 0) or timings.get("load_ms", 0),
        )
        points.append(point)
        print(f"    n={n_predict:4d}: {toks_out} toks out, "
              f"{point.decode_tok_s:.1f} tok/s decode, "
              f"{point.prefill_tok_s:.1f} tok/s prefill")

    return points


def benchmark_reasoning(server: str, port: str, model_name: str) -> list[ReasoningResult]:
    """Run standard reasoning tests."""
    print(f"  [{model_name}] Reasoning...")
    results = []

    # Test 1: Bat and ball
    resp = _post(server, "/v1/chat/completions", {
        "model": model_name,
        "messages": [{"role": "user", "content": (
            "A bat and a ball cost $1.10 together. "
            "The bat costs $1.00 more than the ball. "
            "How much does the ball cost? Give only the dollar amount."
        )}],
        "max_tokens": 50,
        "temperature": 0.1,
    }, timeout=60)

    answer = _extract_content(resp)
    bat_ball = ReasoningResult(
        name="bat_and_ball",
        expected="$0.05",
        answer=answer or "NO_RESPONSE",
        correct=_is_dollar(answer, 0.05),
    )
    results.append(bat_ball)
    print(f"    bat/ball: {bat_ball.answer} (correct={bat_ball.correct})")

    # Test 2: Math — 17 * 23 + 42
    resp = _post(server, "/v1/chat/completions", {
        "model": model_name,
        "messages": [{"role": "user", "content": (
            "What is 17 * 23 + 42? Give only the number."
        )}],
        "max_tokens": 50,
        "temperature": 0.1,
    }, timeout=60)

    answer = _extract_content(resp)
    math_test = ReasoningResult(
        name="math_17x23_plus_42",
        expected="433",
        answer=answer or "NO_RESPONSE",
        correct=_is_number(answer, 433),
    )
    results.append(math_test)
    print(f"    math: {math_test.answer} (correct={math_test.correct})")

    # Test 3: Fibonacci code generation
    resp = _post(server, "/v1/chat/completions", {
        "model": model_name,
        "messages": [{"role": "user", "content": (
            "Write a Python function to compute the nth Fibonacci number iteratively. "
            "Include docstring and type hints."
        )}],
        "max_tokens": 500,
        "temperature": 0.7,
    }, timeout=120)

    answer = _extract_content(resp)
    has_def = "def " in (answer or "")
    has_docstring = '"""' in str(answer) or "'''" in str(answer)
    has_type_hint = "->" in str(answer) if answer else False
    code_ok = has_def and ("fib" in str(answer).lower() or "fibo" in str(answer).lower())

    fib_test = ReasoningResult(
        name="fibonacci_code",
        correct=code_ok,
        answer=f"def={has_def}, docstring={has_docstring}, type_hint={has_type_hint}",
        expected="iterative implementation with docstring and type hints",
    )
    results.append(fib_test)
    print(f"    fibonacci: {'PASS' if code_ok else 'FAIL'} ({fib_test.answer})")

    # Test 4: Sheep riddle (lateral thinking)
    resp = _post(server, "/v1/chat/completions", {
        "model": model_name,
        "messages": [{"role": "user", "content": (
            "A farmer has 17 sheep. All but 9 run away. "
            "How many sheep does the farmer have left? Answer only with the number."
        )}],
        "max_tokens": 20,
        "temperature": 0.1,
    }, timeout=60)

    answer = _extract_content(resp)
    riddle = ReasoningResult(
        name="sheep_riddle",
        expected="9",
        answer=answer or "NO_RESPONSE",
        correct=_is_number(answer, 9),
    )
    results.append(riddle)
    print(f"    sheep: {riddle.answer} (correct={riddle.correct})")

    return results


def benchmark_context_window(server: str, port: str, model_name: str) -> list[ContextResult]:
    """Test context retrieval at increasing lengths."""
    print(f"  [{model_name}] Context window...")
    results = []

    # Test A: 10 items, ask for #7
    items_10 = "\n".join(f"Item number {i}" for i in range(1, 11))
    resp = _post(server, "/v1/chat/completions", {
        "model": model_name,
        "messages": [{"role": "user", "content": (
            f"{items_10}\n\nWhich item is number 7? Answer only with the number."
        )}],
        "max_tokens": 20,
        "temperature": 0.1,
    }, timeout=60)

    answer = _extract_content(resp)
    ctx_10 = ContextResult(
        items_stored=10,
        retrieval_results={"question": "#7 of 10", "answer": answer or "NO_RESPONSE"},
        correct=_is_number(answer, 7),
    )
    results.append(ctx_10)
    print(f"    context #7/10: {ctx_10.retrieval_results['answer']} (correct={ctx_10.correct})")

    # Test B: 50 items, ask for #25
    items_50 = "\n".join(f"Item number {i}" for i in range(1, 51))
    resp = _post(server, "/v1/chat/completions", {
        "model": model_name,
        "messages": [{"role": "user", "content": (
            f"{items_50}\n\nWhich item is number 25? Answer only with the number."
        )}],
        "max_tokens": 20,
        "temperature": 0.1,
    }, timeout=120)

    answer = _extract_content(resp)
    ctx_50 = ContextResult(
        items_stored=50,
        retrieval_results={"question": "#25 of 50", "answer": answer or "NO_RESPONSE"},
        correct=_is_number(answer, 25),
    )
    results.append(ctx_50)
    print(f"    context #25/50: {ctx_50.retrieval_results['answer']} (correct={ctx_50.correct})")

    return results


def benchmark_concurrency(server: str, presets: list[str], output_tok: int = 80) -> list[ConcurrencyResult]:
    """Test concurrency and eviction behavior."""
    print("  [concurrency] Testing...")
    results = []

    # Get ports for loaded models
    ports: dict[str, str] = {}
    for preset in presets:
        port = _get_port(server, preset)
        if port:
            ports[preset] = port
        else:
            print(f"    {preset}: not loaded, skipping")

    if len(ports) < 2:
        results.append(ConcurrencyResult(
            scenario="partial",
            models=list(ports.keys()),
            note=f"Only {len(ports)} models loaded",
        ))
        return results

    # Solo baselines (warm up first)
    solo: dict[str, float] = {}
    for preset, port in ports.items():
        _post(server, f"http://127.0.0.1:{port}/completion", {
            "prompt": "Hi", "n_predict": 4, "temperature": 0.1, "stream": False,
        }, timeout=30)

    for preset, port in ports.items():
        resp = _post(server, f"http://127.0.0.1:{port}/completion", {
            "prompt": "Write a Python function that computes the factorial of a number iteratively.",
            "n_predict": output_tok,
            "temperature": 0.7,
            "stream": False,
        }, timeout=120)
        if resp:
            solo[preset] = resp.get("timings", {}).get("predicted_per_second", 0)

    # Concurrent — fire both at once
    concurrent: dict[str, float] = {}
    import threading

    def _bench(preset: str, port: str, results_dict: dict):
        resp = _post(server, f"http://127.0.0.1:{port}/completion", {
            "prompt": "Write a Python function that computes the factorial of a number iteratively.",
            "n_predict": output_tok,
            "temperature": 0.7,
            "stream": False,
        }, timeout=120)
        if resp:
            results_dict[preset] = resp.get("timings", {}).get("predicted_per_second", 0)

    t1 = threading.Thread(target=_bench, args=(list(ports.keys())[0], list(ports.values())[0], concurrent))
    t2 = threading.Thread(target=_bench, args=(list(ports.keys())[1], list(ports.values())[1], concurrent))
    t1.start()
    t2.start()
    t1.join(timeout=120)
    t2.join(timeout=120)

    degradation: dict[str, float] = {}
    for preset in ports:
        if preset in solo and preset in concurrent and solo[preset] > 0:
            degradation[preset] = round(
                (solo[preset] - concurrent[preset]) / solo[preset] * 100, 1
            )

    scenario_name = "+".join(p.replace("-heretic", "").replace("-uncensored", "") for p in presets[:3])

    results.append(ConcurrencyResult(
        scenario=scenario_name,
        models=list(ports.keys()),
        solo_throughput={k: round(v, 1) for k, v in solo.items()},
        concurrent_throughput={k: round(v, 1) for k, v in concurrent.items()},
        degradation_pct=degradation,
    ))

    # Test eviction: try loading a 3rd model
    print("    Testing eviction...")
    all_presets = _get_all_presets(server)
    third = None
    for p in all_presets:
        if p not in presets and "qwen3.8" in p or "gpt-oss-120b" in p:
            third = p
            break

    if third:
        _post(server, "/models/load", {"model": third}, timeout=60)
        time.sleep(5)
        data = _get(server, "/models")
        loaded_after = []
        evicted = []
        if data:
            for m in data.get("data", []):
                if m["status"]["value"] == "loaded":
                    loaded_after.append(m["id"])
                elif m["id"] not in [p.replace("-", "") for p in presets]:
                    # Was loaded before, now unloaded
                    pass
        results[-1].eviction_rule = "LRU"
        results[-1].max_concurrent = len(loaded_after)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_content(resp: dict | None) -> str | None:
    """Extract text content from a chat completion response."""
    if not resp or "choices" not in resp:
        return None
    choice = resp["choices"][0].get("message", {})
    content = choice.get("content", "")

    # Try reasoning_content for deepseek-style models
    if not content or len(content) < 2:
        rc = choice.get("reasoning_content", "")
        if rc and len(rc) > 10:
            return rc[:200]  # Just grab the start for verification

    return content


def _is_dollar(text: str | None, expected: float) -> bool:
    """Check if text contains a dollar amount matching expected."""
    if not text:
        return False
    nums = re.findall(r'\$?([0-9]+(?:\.[0-9]+)?)', text)
    for n in nums:
        try:
            if abs(float(n) - expected) < 0.01:
                return True
        except ValueError:
            pass
    return False


def _is_number(text: str | None, expected: int) -> bool:
    """Check if text contains a number matching expected."""
    if not text:
        return False
    nums = re.findall(r'-?[0-9]+', text)
    for n in nums:
        try:
            if int(n) == expected:
                return True
        except ValueError:
            pass
    return False


def _get_all_presets(server: str) -> list[str]:
    """Get all available preset names from the server."""
    data = _get(server, "/models")
    if not data:
        return []
    presets = set()
    for m in data.get("data", []):
        name = m.get("id", "")
        if name:
            presets.add(name)
    return sorted(presets)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark(
    server: str,
    preset: str,
    output_path: str | None = None,
    full: bool = False,
) -> BenchmarkRun:
    """Run the full benchmark suite for one model preset."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {preset}")
    print(f"Server: {server}")
    print(f"{'='*60}")

    # Detect hardware
    hw = detect_hardware()

    # Load the model
    print(f"\nLoading {preset}...")
    _post(server, "/models/load", {"model": preset}, timeout=60)
    if not _wait_for_loaded(server, preset):
        print(f"ERROR: {preset} did not load in time")
        return BenchmarkRun(hardware=hw, model=ModelMetadata(name=preset))

    # Get port and model info
    port = _get_port(server, preset)
    if not port:
        print(f"ERROR: Could not get port for {preset}")
        return BenchmarkRun(hardware=hw, model=ModelMetadata(name=preset))

    # Gather model metadata from the router
    data = _get(server, "/models")
    model_meta = ModelMetadata(name=preset)
    if data:
        for m in data.get("data", []):
            if preset.replace("-", "") in m["id"].replace("-", "") or preset in m["id"]:
                args = m["status"].get("args", [])
                preset_name = m.get("id", preset)
                model_meta.name = preset_name
                # Extract model path from args
                try:
                    idx = args.index("--model")
                    model_path = args[idx + 1]
                    model_meta.quantization = Path(model_path).stem.split("-")[-1] if "-" in Path(model_path).stem else ""
                except (ValueError, IndexError):
                    pass
                break

    # Run benchmarks
    throughput = benchmark_throughput(server, port, preset)
    reasoning = benchmark_reasoning(server, port, preset)
    context = benchmark_context_window(server, port, preset)

    # Memory snapshot
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=10
        )
        rss_mb = 0.0
        for line in result.stdout.split("\n"):
            if "llama-server" in line and port in line:
                parts = line.split()
                rss_mb = float(parts[5]) / 1024
                break
    except Exception:
        rss_mb = 0.0

    run = BenchmarkRun(
        hardware=hw,
        model=model_meta,
        throughput_points=throughput,
        reasoning_results=reasoning,
        context_results=context,
        memory_rss_mb=rss_mb,
    )

    if output_path:
        run.save(output_path)

    return run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="halo-benchmarks runner")
    parser.add_argument("--preset", required=True, help="Model preset name")
    parser.add_argument("--server", default="http://127.0.0.1:1244", help="llama-server URL")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--compare", help="Directory of raw JSON files to compare")
    parser.add_argument("--concurrency", nargs="+", metavar="PRESET", help="Test concurrency for given presets")

    args = parser.parse_args(argv)

    if args.compare:
        manifest = ComparisonManifest.from_dir(args.compare)
        table = manifest.comparison_table
        print("\n=== COMPARISON TABLE ===")
        for name, row in table.items():
            print(f"\n{name}:")
            for k, v in row.items():
                print(f"  {k}: {v}")
        return

    if args.concurrency:
        result = benchmark_concurrency(args.server, args.concurrency)
        print(json.dumps([r.to_dict() for r in result], indent=2))
        return

    run = run_benchmark(args.server, args.preset, args.output, full=args.full)
    if not args.output:
        print(json.dumps(run.to_dict(), indent=2))


if __name__ == "__main__":
    main()
