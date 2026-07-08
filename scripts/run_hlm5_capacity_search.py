from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def check_metric(name: str, value: float, op: str, threshold: float) -> dict[str, Any]:
    if op == ">=":
        passed = value >= threshold
    elif op == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"unsupported check op {op}")

    return {
        "name": name,
        "value": value,
        "op": op,
        "threshold": threshold,
        "passed": passed,
    }


def evaluate_metrics(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    checks = [
        check_metric(
            "before_training_accuracy_low",
            float(metrics["before_accuracy"]),
            "<=",
            args.max_before_accuracy,
        ),
        check_metric(
            "trained_accuracy",
            float(metrics["trained_accuracy"]),
            ">=",
            args.min_accuracy,
        ),
        check_metric(
            "edited_accuracy",
            float(metrics["edited_accuracy"]),
            ">=",
            args.min_accuracy,
        ),
        check_metric(
            "remaining_accuracy",
            float(metrics["remaining_accuracy"]),
            ">=",
            args.min_accuracy,
        ),
        check_metric(
            "max_unaffected_probability_shift",
            float(metrics["max_unaffected_probability_shift"]),
            "<=",
            args.max_side_effect,
        ),
        check_metric(
            "trained_crosstalk",
            float(metrics["trained_crosstalk"]),
            "<=",
            args.max_crosstalk,
        ),
        check_metric(
            "unique_target_values",
            float(metrics["unique_target_values"]),
            ">=",
            float(metrics["pairs"]),
        ),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "failed_checks": [check for check in checks if not check["passed"]],
    }


def compact_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"values={metrics['value_count']}, "
        f"unique_targets={metrics['unique_target_values']}, "
        f"before={metrics['before_accuracy']:.3f}, "
        f"trained={metrics['trained_accuracy']:.3f}, "
        f"edited={metrics['edited_accuracy']:.3f}, "
        f"remaining={metrics['remaining_accuracy']:.3f}, "
        f"side_effect={metrics['max_unaffected_probability_shift']:.6f}, "
        f"rho^5={metrics['trained_crosstalk']:.6f}"
    )


def build_train_command(pair_count: int, metrics_path: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "run_train_key_value.py",
        "--init-mode",
        args.init_mode,
        "--pairs",
        str(pair_count),
        "--memory-size",
        str(args.memory_size),
        "--dim",
        str(args.dim),
        "--steps",
        str(args.steps),
        "--lr",
        str(args.lr),
        "--route-weight",
        str(args.route_weight),
        "--coherence-weight",
        str(args.coherence_weight),
        "--coherence-margin",
        str(args.coherence_margin),
        "--slot-noise",
        str(args.slot_noise),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--json-out",
        str(metrics_path),
    ]
    if args.values is not None:
        command.extend(["--values", str(args.values)])
    return command


def run_point(
    pair_count: int,
    metrics_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics_path = metrics_dir / f"random_p{pair_count}.json"
    command = build_train_command(pair_count, metrics_path, args)
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - start

    point: dict[str, Any] = {
        "pairs": pair_count,
        "command": command,
        "returncode": result.returncode,
        "elapsed_sec": elapsed,
        "metrics_path": str(metrics_path),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "command_ok": result.returncode == 0,
        "passed": False,
        "checks": [],
        "failed_checks": [],
    }

    if result.returncode != 0:
        return point

    if not metrics_path.exists():
        point["command_ok"] = False
        point["stderr_tail"] = (
            point["stderr_tail"] + f"\nmissing metrics file: {metrics_path}"
        )
        return point

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    evaluation = evaluate_metrics(metrics, args)
    point.update(
        {
            "metrics": metrics,
            "passed": evaluation["passed"],
            "checks": evaluation["checks"],
            "failed_checks": evaluation["failed_checks"],
        }
    )
    return point


def run_search(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    report_dir = (ROOT / args.report_dir).resolve()
    metrics_dir = report_dir / "capacity_search"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    points: dict[int, dict[str, Any]] = {}

    def evaluate(pair_count: int) -> dict[str, Any]:
        if pair_count not in points:
            print(f"\ncapacity probe: pairs={pair_count}")
            point = run_point(pair_count, metrics_dir, args)
            points[pair_count] = point

            if point["command_ok"] and "metrics" in point:
                status = "PASS" if point["passed"] else "FAIL"
                print(f"  {status}: {compact_metrics(point['metrics'])}")
                for check in point["failed_checks"]:
                    print(
                        "  "
                        f"FAIL: {check['name']} "
                        f"{check['value']:.6f} {check['op']} {check['threshold']:.6f}"
                    )
            else:
                print(f"  ERROR: command returned {point['returncode']}")
                if point["stderr_tail"]:
                    print(point["stderr_tail"])

        return points[pair_count]

    low = args.min_pairs
    high = args.max_pairs

    low_point = evaluate(low)
    if not low_point["command_ok"] or not low_point["passed"]:
        return points

    high_point = evaluate(high)
    if not high_point["command_ok"] or high_point["passed"]:
        return points

    pass_low = low
    fail_high = high
    while pass_low + 1 < fail_high:
        mid = (pass_low + fail_high) // 2
        mid_point = evaluate(mid)
        if not mid_point["command_ok"]:
            break
        if mid_point["passed"]:
            pass_low = mid
        else:
            fail_high = mid

    return points


def summarize(
    points: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ordered = [points[pair_count] for pair_count in sorted(points)]
    command_errors = [point for point in ordered if not point["command_ok"]]
    passing = [point for point in ordered if point["command_ok"] and point["passed"]]
    failing = [point for point in ordered if point["command_ok"] and not point["passed"]]

    max_passing_pairs = max((point["pairs"] for point in passing), default=None)
    first_failing_pairs = min((point["pairs"] for point in failing), default=None)
    if max_passing_pairs is not None:
        higher_failures = [
            point["pairs"]
            for point in failing
            if point["pairs"] > max_passing_pairs
        ]
        first_failing_pairs = min(higher_failures, default=first_failing_pairs)

    return {
        "search_completed": not command_errors,
        "monotonic_assumption": (
            "binary search assumes pass/fail is monotonic with pair count "
            "for fixed seed and config"
        ),
        "max_passing_pairs": max_passing_pairs,
        "first_failing_pairs": first_failing_pairs,
        "config": {
            "min_pairs": args.min_pairs,
            "max_pairs": args.max_pairs,
            "memory_size": args.memory_size,
            "value_count": args.values,
            "value_policy": "fixed" if args.values is not None else "auto=max(32,pairs)",
            "dim": args.dim,
            "steps": args.steps,
            "lr": args.lr,
            "route_weight": args.route_weight,
            "coherence_weight": args.coherence_weight,
            "coherence_margin": args.coherence_margin,
            "init_mode": args.init_mode,
            "slot_noise": args.slot_noise,
            "seed": args.seed,
            "device": args.device,
        },
        "thresholds": {
            "min_accuracy": args.min_accuracy,
            "max_before_accuracy": args.max_before_accuracy,
            "max_side_effect": args.max_side_effect,
            "max_crosstalk": args.max_crosstalk,
        },
        "points": ordered,
    }


def write_reports(report: dict[str, Any], report_dir: Path) -> None:
    json_path = report_dir / "hlm5_capacity_search_report.json"
    text_path = report_dir / "hlm5_capacity_search_report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "HLM5 Capacity Boundary Search",
        f"search_completed: {report['search_completed']}",
        f"max_passing_pairs: {report['max_passing_pairs']}",
        f"first_failing_pairs: {report['first_failing_pairs']}",
        "",
        "Config:",
    ]
    config = report["config"]
    lines.extend(
        [
            (
                f"range={config['min_pairs']}..{config['max_pairs']}, "
                f"memory_size={config['memory_size']}, "
                f"values={config['value_count'] or config['value_policy']}"
            ),
            (
                f"dim={config['dim']}, steps={config['steps']}, "
                f"seed={config['seed']}, init_mode={config['init_mode']}, "
                f"device={config['device']}"
            ),
            (
                f"route_weight={config['route_weight']}, "
                f"coherence_weight={config['coherence_weight']}, "
                f"coherence_margin={config['coherence_margin']}"
            ),
            "",
            "Points:",
        ]
    )

    for point in report["points"]:
        if not point["command_ok"]:
            lines.append(f"- ERROR: p{point['pairs']} returncode={point['returncode']}")
            continue
        status = "PASS" if point["passed"] else "FAIL"
        lines.append(f"- {status}: p{point['pairs']} {compact_metrics(point['metrics'])}")
        for check in point["failed_checks"]:
            lines.append(
                "  "
                f"FAIL: {check['name']} "
                f"{check['value']:.6f} {check['op']} {check['threshold']:.6f}"
            )

    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwrote report: {text_path}")
    print(f"wrote report: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binary-search the editable capacity boundary for HLM5 key-value training."
    )
    parser.add_argument("--min-pairs", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--memory-size", type=int, default=128)
    parser.add_argument("--values", type=int, default=None)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--route-weight", type=float, default=2.0)
    parser.add_argument("--coherence-weight", type=float, default=0.2)
    parser.add_argument("--coherence-margin", type=float, default=0.25)
    parser.add_argument(
        "--init-mode",
        choices=["random", "mismatch", "noisy", "exact"],
        default="random",
    )
    parser.add_argument("--slot-noise", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="training device passed to run_train_key_value.py",
    )
    parser.add_argument("--report-dir", type=str, default="reports")
    parser.add_argument("--min-accuracy", type=float, default=0.99)
    parser.add_argument("--max-before-accuracy", type=float, default=0.25)
    parser.add_argument("--max-side-effect", type=float, default=0.01)
    parser.add_argument("--max-crosstalk", type=float, default=0.01)
    args = parser.parse_args()

    if args.max_pairs is None:
        args.max_pairs = args.memory_size
    if args.min_pairs < 4:
        parser.error("--min-pairs must be at least 4 because the edit test uses pair index 3")
    if args.max_pairs < args.min_pairs:
        parser.error("--max-pairs must be >= --min-pairs")
    if args.memory_size < args.max_pairs:
        parser.error("--memory-size must be >= --max-pairs")
    if args.values is not None and args.values <= 0:
        parser.error("--values must be positive")

    print("HLM5 capacity boundary search")
    print(f"root={ROOT}")
    print(
        f"range={args.min_pairs}..{args.max_pairs}, "
        f"memory_size={args.memory_size}, steps={args.steps}"
    )

    points = run_search(args)
    report = summarize(points, args)
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    write_reports(report, report_dir)

    if not report["search_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
