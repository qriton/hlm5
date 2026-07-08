from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def run_command(name: str, command: list[str]) -> dict[str, Any]:
    print("\n" + "=" * 88)
    print(f"HLM5 SCALE STEP: {name}")
    print("=" * 88)
    print("COMMAND:", " ".join(command))
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.perf_counter() - start
    print(f"DONE: {name} ({elapsed:.1f}s, returncode={result.returncode})")
    return {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "elapsed_sec": elapsed,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def capacity_specs(profile: str) -> list[tuple[int, int, int, int]]:
    # (memory_size, dim, min_pairs, max_pairs). dim scales with memory because the
    # binding bar is key geometry (rho^5 over active keys), which packs ~1 pair per
    # key DIMENSION, not per slot — the first scale run held dim=64 while testing
    # 96/192 pairs and failed only the crosstalk bar (recall/edit stayed perfect).
    # The (128, 64) probe holds dim while doubling slots: if the boundary stays where
    # m=64/d=64 put it (~64), the wall follows dim, not slot count.
    if profile == "smoke":
        return [(64, 64, 32, 64)]
    if profile == "standard":
        return [(128, 128, 64, 128), (256, 256, 128, 256), (128, 64, 64, 96)]
    if profile == "large":
        return [
            (128, 128, 64, 128),
            (256, 256, 128, 256),
            (128, 64, 64, 96),
            (512, 512, 256, 512),
        ]
    raise ValueError(f"unknown profile {profile}")


def kb_specs(profile: str) -> list[dict[str, Any]]:
    if profile == "smoke":
        return [
            {
                "contexts": 64,
                "vocab": 128,
                "dim": 128,
                "steps": 500,
                "ks": "8,16,32,48,64",
            }
        ]
    if profile == "standard":
        return [
            {
                "contexts": 128,
                "vocab": 256,
                "dim": 256,
                "steps": 2500,
                "ks": "16,32,64,96,128",
            },
            {
                "contexts": 256,
                "vocab": 512,
                "dim": 512,
                "steps": 3000,
                "ks": "32,64,96,128,160,192,224,256",
            },
        ]
    if profile == "large":
        return [
            {
                "contexts": 256,
                "vocab": 512,
                "dim": 512,
                "steps": 4000,
                "ks": "32,64,96,128,160,192,224,256",
            },
            {
                "contexts": 512,
                "vocab": 1024,
                "dim": 512,
                "steps": 5000,
                "ks": "64,128,192,256,320,384,448,512",
            },
        ]
    raise ValueError(f"unknown profile {profile}")


def summarize_capacity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_size": report["config"]["memory_size"],
        "dim": report["config"]["dim"],
        "max_passing_pairs": report["max_passing_pairs"],
        "first_failing_pairs": report["first_failing_pairs"],
        "points": [
            {
                "pairs": point["pairs"],
                "passed": point["passed"],
                "rho5": point.get("metrics", {}).get("trained_crosstalk"),
                "side_effect": point.get("metrics", {}).get(
                    "max_unaffected_probability_shift"
                ),
            }
            for point in report["points"]
        ],
    }


def summarize_kb(report: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    perfect = [
        row["K"]
        for row in report["sweep"]
        if row["inject_flip"] == 1
        and row["audit_correct"] == 1
        and row["others_intact"] in (1, None)  # K==N: no others left, vacuously intact
    ]
    return {
        "contexts": spec["contexts"],
        "vocab": spec["vocab"],
        "dim": spec["dim"],
        "steps": spec["steps"],
        "base_acc": report["base_acc"],
        "perfect_capacity": max(perfect, default=0),
        "sweep": report["sweep"],
    }


def write_report(report_dir: Path, report: dict[str, Any]) -> None:
    json_path = report_dir / "hlm5_scale_suite_report.json"
    text_path = report_dir / "hlm5_scale_suite_report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "HLM5 Scale Suite",
        f"overall: {'PASS' if report['passed'] else 'FAIL'}",
        f"profile: {report['profile']}",
        f"device: {report['device']}",
        "",
        "Capacity Boundary:",
    ]
    for item in report["capacity"]:
        lines.append(
            "- "
            f"memory={item['memory_size']} dim={item['dim']}: "
            f"max_pass={item['max_passing_pairs']}, "
            f"first_fail={item['first_failing_pairs']}"
        )

    lines.extend(["", "Editable KB Injection:"])
    for item in report["kb_injection"]:
        lines.append(
            "- "
            f"dim={item['dim']}, contexts={item['contexts']}: "
            f"base_acc={item['base_acc']:.3f}, "
            f"perfect_capacity={item['perfect_capacity']}"
        )

    lines.extend(["", "Steps:"])
    for step in report["steps"]:
        status = "PASS" if step["returncode"] == 0 else "FAIL"
        lines.append(
            f"- {status}: {step['name']} "
            f"(returncode={step['returncode']}, {step['elapsed_sec']:.1f}s)"
        )

    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwrote report: {text_path}")
    print(f"wrote report: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HLM5 scaled training/evaluation suite.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "standard", "large"],
        default="standard",
    )
    parser.add_argument("--report-dir", type=str, default="reports/scale")
    parser.add_argument("--capacity-steps", type=int, default=1000)
    parser.add_argument("--kb-steps-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-capacity", action="store_true")
    parser.add_argument("--skip-kb", action="store_true")
    args = parser.parse_args()

    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    print("HLM5 scale suite")
    print(f"root={ROOT}")
    print(f"profile={args.profile}")
    print(f"report_dir={report_dir}")

    steps: list[dict[str, Any]] = []
    capacity_reports: list[dict[str, Any]] = []
    kb_reports: list[dict[str, Any]] = []

    if not args.skip_capacity:
        for memory_size, dim, min_pairs, max_pairs in capacity_specs(args.profile):
            capacity_dir = report_dir / f"capacity_m{memory_size}_d{dim}"
            command = [
                sys.executable,
                "run_hlm5_capacity_search.py",
                "--min-pairs",
                str(min_pairs),
                "--max-pairs",
                str(max_pairs),
                "--memory-size",
                str(memory_size),
                "--dim",
                str(dim),
                "--steps",
                str(args.capacity_steps),
                "--device",
                args.device,
                "--report-dir",
                str(capacity_dir),
            ]
            step = run_command(f"capacity search memory={memory_size} dim={dim}", command)
            steps.append(step)
            report_path = capacity_dir / "hlm5_capacity_search_report.json"
            # returncode gate: don't summarize a stale report left by a previous run
            if step["returncode"] == 0 and report_path.exists():
                capacity_reports.append(summarize_capacity(load_json(report_path)))

    if not args.skip_kb:
        for spec in kb_specs(args.profile):
            scaled_steps = max(1, int(spec["steps"] * args.kb_steps_scale))
            out_path = (
                report_dir
                / f"lm_kb_ctx{spec['contexts']}_d{spec['dim']}_steps{scaled_steps}.json"
            )
            command = [
                sys.executable,
                "run_lm_kb_inject.py",
                "--contexts",
                str(spec["contexts"]),
                "--vocab",
                str(spec["vocab"]),
                "--dim",
                str(spec["dim"]),
                "--steps",
                str(scaled_steps),
                "--boost",
                "auto",
                "--ks",
                spec["ks"],
                "--out",
                str(out_path),
            ]
            step = run_command(
                f"LM KB injection contexts={spec['contexts']} dim={spec['dim']}",
                command,
            )
            steps.append(step)
            # returncode gate: don't summarize a stale report left by a previous run
            if step["returncode"] == 0 and out_path.exists():
                summarized = dict(spec)
                summarized["steps"] = scaled_steps
                kb_reports.append(summarize_kb(load_json(out_path), summarized))

    report = {
        "passed": all(step["returncode"] == 0 for step in steps),
        "profile": args.profile,
        "device": args.device,
        "capacity": capacity_reports,
        "kb_injection": kb_reports,
        "steps": steps,
    }
    write_report(report_dir, report)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
