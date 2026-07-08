"""Self-containment verification for hlm5-release/results/.

Loads each externally-sourced JSON artifact copied into results/ and asserts
the load-bearing numbers the paper cites against them. Prints one PASS/FAIL
line per artifact and exits nonzero if any artifact fails.

Run from repo root: python scripts/verify_artifacts.py
"""
from __future__ import annotations

import json
import math
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

FAILURES: list[str] = []


def load(name: str):
    path = os.path.join(RESULTS, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f" -- {detail}"
    print(line)
    if not condition:
        FAILURES.append(label)


def verify_capacity_search_report() -> None:
    d = load("hlm5_capacity_search_report.json")
    val = d.get("max_passing_pairs")
    check(
        "hlm5_capacity_search_report.json: max_passing_pairs == 105",
        val == 105,
        f"asserted max_passing_pairs == 105, found {val!r}",
    )


def verify_scale_suite_report() -> None:
    d = load("hlm5_scale_suite_report.json")
    capacity = d.get("capacity", [])

    def find(memory_size: int, dim: int):
        for entry in capacity:
            if entry.get("memory_size") == memory_size and entry.get("dim") == dim:
                return entry
        return None

    # The 95/128 point: memory_size=128, dim=64 saturates below the memory's
    # nominal slot count (max_passing_pairs=95 of 128 slots).
    e_95 = find(128, 64)
    ok_95 = e_95 is not None and e_95.get("max_passing_pairs") == 95
    check(
        "hlm5_scale_suite_report.json: capacity point M=128,D=64 max_passing_pairs == 95 (the 95/128 point)",
        ok_95,
        f"asserted capacity[memory_size=128,dim=64].max_passing_pairs == 95, found {e_95.get('max_passing_pairs') if e_95 else 'ENTRY NOT FOUND'!r}",
    )

    # Saturation points: M=128,D=128 -> 128 and M=256,D=256 -> 256.
    e_128 = find(128, 128)
    ok_128 = e_128 is not None and e_128.get("max_passing_pairs") == 128
    check(
        "hlm5_scale_suite_report.json: saturation point M=128,D=128 max_passing_pairs == 128",
        ok_128,
        f"asserted capacity[memory_size=128,dim=128].max_passing_pairs == 128, found {e_128.get('max_passing_pairs') if e_128 else 'ENTRY NOT FOUND'!r}",
    )

    e_256 = find(256, 256)
    ok_256 = e_256 is not None and e_256.get("max_passing_pairs") == 256
    check(
        "hlm5_scale_suite_report.json: saturation point M=256,D=256 max_passing_pairs == 256",
        ok_256,
        f"asserted capacity[memory_size=256,dim=256].max_passing_pairs == 256, found {e_256.get('max_passing_pairs') if e_256 else 'ENTRY NOT FOUND'!r}",
    )


def verify_incontext_mqar() -> None:
    d = load("incontext_mqar.json")

    def find(arm: str, num_pairs: int):
        for entry in d:
            if entry.get("arm") == arm and entry.get("num_pairs") == num_pairs:
                return entry.get("accuracy")
        return None

    t_only = find("transformer-only", 64)
    ok_t = t_only is not None and close(t_only, 0.81640625, 1e-4)
    check(
        "incontext_mqar.json: transformer-only accuracy @ 64 pairs ~= 0.81640625",
        ok_t,
        f"asserted ~=0.81640625 tol 1e-4, found {t_only!r}",
    )

    t_hlm5 = find("transformer+HLM5", 64)
    ok_h = t_hlm5 is not None and close(t_hlm5, 0.11796875, 1e-4)
    check(
        "incontext_mqar.json: transformer+HLM5 accuracy @ 64 pairs ~= 0.11796875",
        ok_h,
        f"asserted ~=0.11796875 tol 1e-4, found {t_hlm5!r}",
    )


def verify_g2c_rare() -> None:
    d = load("g2c_rare.json")
    val = d.get("reachable_fraction")
    ok = val is not None and close(val, 0.01849, 1e-4)
    check(
        "g2c_rare.json: reachable_fraction ~= 0.01849 (tol 1e-4)",
        ok,
        f"asserted ~=0.01849 tol 1e-4, found {val!r}",
    )


def verify_train_summaries() -> None:
    b = load("baseline_train_summary.json")
    h = load("hybrid_train_summary.json")
    check(
        "baseline_train_summary.json: steps == 153000",
        b.get("steps") == 153000,
        f"asserted steps == 153000, found {b.get('steps')!r}",
    )
    check(
        "hybrid_train_summary.json: steps == 153000",
        h.get("steps") == 153000,
        f"asserted steps == 153000, found {h.get('steps')!r}",
    )


def verify_edit_receipts() -> None:
    d = load("edit_receipts.json")
    checks = {
        "receipts_verified": True,
        "tamper_value_detected": True,
        "tamper_prediction_detected": True,
        "tamper_prompt_detected": True,
    }
    ok = all(d.get(k) == v for k, v in checks.items())
    detail = ", ".join(f"{k}={d.get(k)!r}" for k in checks)
    check(
        "edit_receipts.json: receipts_verified and all tamper_*_detected flags == true",
        ok,
        detail,
    )


def verify_lm_kb_inject() -> None:
    d = load("lm_kb_inject.json")
    sweep = {entry.get("K"): entry.get("inject_flip") for entry in d.get("sweep", [])}
    base_acc = d.get("base_acc")
    expected = {16: 1.0, 32: 1.0, 48: 0.97917, 64: 0.984375, 80: 0.9875}

    ok_base = base_acc is not None and close(base_acc, 1.0, 1e-4)
    check(
        "lm_kb_inject.json (D=64): base_acc ~= 1.0",
        ok_base,
        f"asserted base_acc ~=1.0 tol 1e-4, found {base_acc!r}",
    )

    for k, exp in expected.items():
        val = sweep.get(k)
        ok = val is not None and close(val, exp, 1e-4)
        check(
            f"lm_kb_inject.json (D=64): inject_flip @ K={k} ~= {exp}",
            ok,
            f"asserted ~={exp} tol 1e-4, found {val!r}",
        )


def verify_no_tax(name: str) -> None:
    d = load(name)
    baseline = d.get("baseline_val_ppl")
    hybrid = d.get("hybrid_val_ppl")
    ok = baseline is not None and hybrid is not None
    check(
        f"{name}: loadable, contains baseline/hybrid PPL fields",
        ok,
        f"baseline_val_ppl={baseline!r}, hybrid_val_ppl={hybrid!r}, delta_pct={d.get('delta_pct')!r}",
    )


def main() -> int:
    verify_capacity_search_report()
    verify_scale_suite_report()
    verify_incontext_mqar()
    verify_g2c_rare()
    verify_train_summaries()
    verify_edit_receipts()
    verify_lm_kb_inject()
    verify_no_tax("g2a_no_tax.json")
    verify_no_tax("g3a_no_tax.json")

    # Count PASS/FAIL lines were already printed; just summarize failures.
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("RESULT: all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
