"""Build outcome-blind HWU64 populations for the registered SCA-2 assay."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .contract import (
    DATASET_GIT_COMMIT,
    DATASET_SCRIPTS_GIT_COMMIT,
    DEFAULT_DATA_ROOT,
    DEVELOPMENT_FOLD,
    GROUP_INTENT_COUNT,
    GROUP_NAMES,
    INTENT_ORDER_SALT,
    MANIFEST_PATH,
    ROW_ORDER_SALT,
    ROWS_PER_INTENT,
    SUPPORT_PER_INTENT,
    TEST_FOLD,
    TRAIN_FOLDS,
    fold_test_relative_path,
    sha256_bytes,
    stable_sha256,
    verify_data_files,
    write_json,
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("empty normalized utterance")
    return normalized


def _row_id(fold: int, source_index: int, normalized: str, intent: str) -> str:
    payload = "\0".join((str(fold), str(source_index), normalized, intent))
    return sha256_bytes(payload.encode("utf-8"))


def read_fold(data_root: Path, fold: int) -> list[dict[str, Any]]:
    path = data_root / fold_test_relative_path(fold)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"rasa_nlu_data"}:
        raise ValueError(f"unexpected HWU64 root keys in {path}")
    data = payload["rasa_nlu_data"]
    examples = data.get("common_examples")
    if not isinstance(examples, list):
        raise ValueError(f"missing common_examples in {path}")

    rows: list[dict[str, Any]] = []
    for source_index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"non-object example at {path}:{source_index}")
        text = example.get("text")
        intent = example.get("intent")
        if not isinstance(text, str) or not isinstance(intent, str):
            raise ValueError(f"invalid example at {path}:{source_index}")
        normalized = normalize_text(text)
        intent = intent.strip()
        if not intent:
            raise ValueError(f"empty intent at {path}:{source_index}")
        rows.append(
            {
                "row_id": _row_id(fold, source_index, normalized, intent),
                "fold": fold,
                "source_index": source_index,
                "intent": intent,
                "utterance": text,
                "normalized": normalized,
                "utterance_sha256": sha256_bytes(normalized.encode("utf-8")),
            }
        )
    return rows


def clean_partition(
    rows: Iterable[dict[str, Any]],
    blocked_normalized: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = list(rows)
    blocked_normalized = blocked_normalized or set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["normalized"]].append(row)

    clean: list[dict[str, Any]] = []
    conflict_rows = 0
    duplicate_rows = 0
    overlap_rows = 0
    for normalized in sorted(grouped):
        group = grouped[normalized]
        if normalized in blocked_normalized:
            overlap_rows += len(group)
            continue
        if len({row["intent"] for row in group}) != 1:
            conflict_rows += len(group)
            continue
        ordered = sorted(group, key=lambda row: row["row_id"])
        clean.append(ordered[0])
        duplicate_rows += len(ordered) - 1
    clean.sort(key=lambda row: row["row_id"])
    return clean, {
        "raw_rows": len(rows),
        "clean_rows": len(clean),
        "conflict_rows_dropped": conflict_rows,
        "duplicate_rows_dropped": duplicate_rows,
        "overlap_rows_dropped": overlap_rows,
    }


def _salted_rank(value: str, salt: str) -> tuple[str, str]:
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest(), value


def select_rows(
    rows: Iterable[dict[str, Any]], count: int, salt: str
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: _salted_rank(row["row_id"], salt))
    if len(ordered) < count:
        raise ValueError(f"requested {count} rows from a population of {len(ordered)}")
    return ordered[:count]


def assign_intent_groups(intents: Iterable[str]) -> dict[str, list[str]]:
    ordered = sorted(intents, key=lambda item: _salted_rank(item, INTENT_ORDER_SALT))
    needed = len(GROUP_NAMES) * GROUP_INTENT_COUNT
    if len(ordered) < needed:
        raise ValueError(f"need {needed} eligible intents, found {len(ordered)}")
    groups = {
        name: ordered[index * GROUP_INTENT_COUNT : (index + 1) * GROUP_INTENT_COUNT]
        for index, name in enumerate(GROUP_NAMES)
    }
    groups["unused_eligible"] = ordered[needed:]
    return groups


def row_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row["row_id"],
        "fold": row["fold"],
        "source_index": row["source_index"],
        "intent": row["intent"],
        "utterance_sha256": row["utterance_sha256"],
    }


def population_sha256(rows: Iterable[dict[str, Any]]) -> str:
    return stable_sha256([row_descriptor(row) for row in rows])


def _rows_for_intents(
    rows: Iterable[dict[str, Any]],
    intents: Iterable[str],
    count_per_intent: int,
    population_name: str,
) -> list[dict[str, Any]]:
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_intent[row["intent"]].append(row)
    selected: list[dict[str, Any]] = []
    for intent in sorted(intents):
        selected.extend(
            select_rows(
                by_intent[intent],
                count_per_intent,
                f"{ROW_ORDER_SALT}\0{population_name}\0{intent}",
            )
        )
    return selected


def _population_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"count": len(rows), "sha256": population_sha256(rows)}


def build_populations(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    folds = {fold: read_fold(data_root, fold) for fold in range(1, 11)}
    raw_development = folds[DEVELOPMENT_FOLD]
    raw_test = folds[TEST_FOLD]
    raw_train = [row for fold in TRAIN_FOLDS for row in folds[fold]]

    development, development_cleanup = clean_partition(raw_development)
    development_norms = {row["normalized"] for row in raw_development}
    test, test_cleanup = clean_partition(raw_test, development_norms)
    heldout_norms = development_norms | {row["normalized"] for row in raw_test}
    train, train_cleanup = clean_partition(raw_train, heldout_norms)

    train_counts = Counter(row["intent"] for row in train)
    development_counts = Counter(row["intent"] for row in development)
    test_counts = Counter(row["intent"] for row in test)
    all_intents = sorted(set(train_counts) | set(development_counts) | set(test_counts))
    eligible = sorted(
        intent
        for intent in all_intents
        if train_counts[intent] >= SUPPORT_PER_INTENT
        and development_counts[intent] >= ROWS_PER_INTENT
        and test_counts[intent] >= ROWS_PER_INTENT
    )
    groups = assign_intent_groups(eligible)
    assigned = {intent for name in GROUP_NAMES for intent in groups[name]}
    if len(assigned) != len(GROUP_NAMES) * GROUP_INTENT_COUNT:
        raise AssertionError("intent groups are not pairwise disjoint")

    target_intents = set(groups["target_supported"])
    source_fit = [row for row in train if row["intent"] not in target_intents]
    populations = {
        "source_fit": source_fit,
        "calibration_support": _rows_for_intents(
            train,
            groups["calibration_supported"],
            SUPPORT_PER_INTENT,
            "calibration-support",
        ),
        "calibration_positive": _rows_for_intents(
            development,
            groups["calibration_supported"],
            ROWS_PER_INTENT,
            "calibration-positive",
        ),
        "calibration_negative": _rows_for_intents(
            development,
            groups["calibration_negative"],
            ROWS_PER_INTENT,
            "calibration-negative",
        ),
        "audit_support": _rows_for_intents(
            train,
            groups["audit_supported"],
            SUPPORT_PER_INTENT,
            "audit-support",
        ),
        "audit_positive": _rows_for_intents(
            development,
            groups["audit_supported"],
            ROWS_PER_INTENT,
            "audit-positive",
        ),
        "audit_negative": _rows_for_intents(
            development,
            groups["audit_negative"],
            ROWS_PER_INTENT,
            "audit-negative",
        ),
        "target_support": _rows_for_intents(
            train,
            groups["target_supported"],
            SUPPORT_PER_INTENT,
            "target-support",
        ),
        "target_development": _rows_for_intents(
            development,
            groups["target_supported"],
            ROWS_PER_INTENT,
            "target-development",
        ),
        "target_test": _rows_for_intents(
            test,
            groups["target_supported"],
            ROWS_PER_INTENT,
            "target-test",
        ),
        "off_support_test": _rows_for_intents(
            test,
            groups["audit_negative"],
            ROWS_PER_INTENT,
            "off-support-test",
        ),
    }

    development_names = (
        "source_fit",
        "calibration_support",
        "calibration_positive",
        "calibration_negative",
        "audit_support",
        "audit_positive",
        "audit_negative",
        "target_support",
        "target_development",
    )
    development_ids = {
        row["row_id"] for name in development_names for row in populations[name]
    }
    test_ids = {
        row["row_id"]
        for name in ("target_test", "off_support_test")
        for row in populations[name]
    }
    if development_ids & test_ids:
        raise AssertionError("test row entered a development population")
    if any(
        row["fold"] == TEST_FOLD
        for name in development_names
        for row in populations[name]
    ):
        raise AssertionError("test fold entered a development population")
    if any(row["intent"] in target_intents for row in source_fit):
        raise AssertionError("target intent entered source covariance fitting")

    return {
        "rows": populations,
        "metadata": {
            "cleanup": {
                "development": development_cleanup,
                "test": test_cleanup,
                "train": train_cleanup,
            },
            "raw_fold_counts": {str(fold): len(folds[fold]) for fold in folds},
            "clean_intent_counts": {
                "train": dict(sorted(train_counts.items())),
                "development": dict(sorted(development_counts.items())),
                "test": dict(sorted(test_counts.items())),
            },
            "all_intents": all_intents,
            "eligible_intents": eligible,
            "ineligible_intents": sorted(set(all_intents) - set(eligible)),
            "groups": groups,
            "test_utterances_tokenized_or_encoded": False,
        },
    }


def make_manifest(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    file_hashes = verify_data_files(data_root)
    built = build_populations(data_root)
    populations = built["rows"]
    manifest = {
        "schema": "hlm5-sca2-hwu64-development-manifest-v1",
        "source": {
            "name": "HWU64 / NLU-Evaluation-Data",
            "root": str(data_root),
            "official_repository": "https://github.com/xliuhw/NLU-Evaluation-Data",
            "official_scripts_repository": "https://github.com/xliuhw/NLU-Evaluation-Scripts",
            "dataset_git_commit": DATASET_GIT_COMMIT,
            "dataset_scripts_git_commit": DATASET_SCRIPTS_GIT_COMMIT,
            "license": "CC BY 4.0",
            "files": file_hashes,
        },
        "constants": {
            "development_fold": DEVELOPMENT_FOLD,
            "test_fold": TEST_FOLD,
            "train_folds": list(TRAIN_FOLDS),
            "support_per_intent": SUPPORT_PER_INTENT,
            "rows_per_intent": ROWS_PER_INTENT,
            "group_intent_count": GROUP_INTENT_COUNT,
            "group_names": list(GROUP_NAMES),
            "intent_order_salt": INTENT_ORDER_SALT,
            "row_order_salt": ROW_ORDER_SALT,
        },
        "metadata": built["metadata"],
        "populations": {
            name: _population_record(rows) for name, rows in populations.items()
        },
        "support_rows": {
            name: [row_descriptor(row) for row in populations[name]]
            for name in ("calibration_support", "audit_support", "target_support")
        },
        "development_hidden_population_names": [
            "source_fit",
            "calibration_support",
            "calibration_positive",
            "calibration_negative",
            "audit_support",
            "audit_positive",
            "audit_negative",
            "target_support",
            "target_development",
        ],
        "test_utterances_tokenized_or_encoded": False,
    }
    manifest["scientific_sha256"] = stable_sha256(manifest)
    return manifest


def verify_manifest(
    data_root: Path = DEFAULT_DATA_ROOT,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    expected = make_manifest(data_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("HWU64 development manifest does not match source files")
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = make_manifest(args.data_root)
    write_json(args.out, manifest)
    print(f"wrote {args.out}")
    print(f"scientific_sha256={manifest['scientific_sha256']}")
    for name, population in manifest["populations"].items():
        print(f"{name}={population['count']}")


if __name__ == "__main__":
    main()

