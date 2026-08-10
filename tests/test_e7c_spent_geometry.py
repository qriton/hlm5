from __future__ import annotations

import runpy
from pathlib import Path

import torch

from hlm5.e7b_runtime import geometry_rows


SCRIPT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "analyze_e7c_spent_geometry.py")
)
CANDIDATE_SPECS = SCRIPT["CANDIDATE_SPECS"]
choose_candidate = SCRIPT["choose_candidate"]
covariance_basis = SCRIPT["covariance_basis"]
directions_for_candidate = SCRIPT["directions_for_candidate"]
geometry_for_directions = SCRIPT["geometry_for_directions"]
normalize_rows = SCRIPT["normalize_rows"]
split_counts = SCRIPT["split_counts"]
split_name = SCRIPT["split_name"]
validation_passes = SCRIPT["validation_passes"]


def test_split_is_deterministic_and_exhaustive() -> None:
    target_ids = list(range(100, 180))
    first = [split_name(target_id) for target_id in target_ids]
    second = [split_name(target_id) for target_id in target_ids]
    assert first == second
    assert set(first) == {"development", "validation"}
    counts = split_counts(target_ids)
    assert sum(counts.values()) == len(target_ids)


def test_normalize_rows_uses_registered_denominator() -> None:
    rows = torch.tensor([[3.0, 4.0], [5.0, 12.0]], dtype=torch.float64)
    expected = rows / (rows.norm(dim=1, keepdim=True) + 1e-12)
    assert torch.equal(normalize_rows(rows), expected)


def test_raw_geometry_matches_e7b_helper() -> None:
    head64 = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-0.5, -0.25, 0.1],
        ],
        dtype=torch.float64,
    )
    hidden = torch.tensor([0.2, -0.1, 0.05], dtype=torch.bfloat16)
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    expected, directions, _ = geometry_rows(head64, hidden, targets)
    actual = geometry_for_directions(head64, hidden, targets, directions)
    fields = (
        "target_id",
        "L",
        "U",
        "beta",
        "float64_margin",
        "geometry_class",
        "hard_blocker",
        "hard_blocker_id",
    )
    assert [tuple(row.get(field) for field in fields) for row in actual] == [
        tuple(row.get(field) for field in fields) for row in expected
    ]


def test_candidate_directions_are_finite_unit_rows() -> None:
    torch.manual_seed(4)
    head = torch.randn(24, 6, dtype=torch.float32)
    head64 = head.to(torch.float64)
    targets = torch.tensor([1, 5, 9, 13], dtype=torch.long)
    mean, eigenvalues, eigenvectors, ridge = covariance_basis(head)
    for spec in CANDIDATE_SPECS:
        directions = directions_for_candidate(
            head64,
            targets,
            spec,
            mean32=mean,
            eigenvalues32=eigenvalues,
            eigenvectors32=eigenvectors,
            ridge=ridge,
        )
        assert directions.shape == (4, 6)
        assert torch.isfinite(directions).all()
        assert torch.allclose(
            directions.norm(dim=1),
            torch.ones(4, dtype=torch.float64),
            atol=1e-10,
            rtol=0,
        )


def test_selection_reads_development_count_and_uses_frozen_tie_order() -> None:
    rows = [
        {"name": "first", "development": {"geometrically_reachable": 8}},
        {"name": "second", "development": {"geometrically_reachable": 9}},
        {"name": "third", "development": {"geometrically_reachable": 9}},
    ]
    assert choose_candidate(rows) == "second"


def test_validation_bars_use_exact_integer_arithmetic() -> None:
    assert validation_passes(
        {"target_count": 100, "geometrically_reachable": 80, "native_admitted": 80}
    )
    assert not validation_passes(
        {"target_count": 100, "geometrically_reachable": 79, "native_admitted": 79}
    )
    assert not validation_passes(
        {"target_count": 100, "geometrically_reachable": 100, "native_admitted": 98}
    )
