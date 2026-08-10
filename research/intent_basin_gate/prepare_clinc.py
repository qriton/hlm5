"""Build outcome-blind CLINC development and test manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from contract import (
    DEVELOPMENT_MANIFEST_PATH,
    EXPERIMENT_ID,
    SCIENTIFIC_CONSTANTS,
    SOURCE_COMMIT,
    SOURCE_DATA_SHA256,
    SOURCE_REPOSITORY,
    TEST_MANIFEST_PATH,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def repository_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_source(source_path: Path) -> dict[str, list[list[str]]]:
    value = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CLINC source must be a JSON object")
    required = {"train", "val", "test", "oos_train", "oos_val", "oos_test"}
    if set(value) != required:
        raise ValueError(f"unexpected CLINC split keys: {sorted(value)}")
    return value


def validate_row(row: object, split: str, index: int) -> tuple[str, str]:
    if not isinstance(row, list) or len(row) != 2:
        raise ValueError(f"{split}[{index}] is not [text, label]")
    text, label = row
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{split}[{index}] has invalid text")
    if not isinstance(label, str) or not label:
        raise ValueError(f"{split}[{index}] has invalid label")
    return text, label


def make_rows(
    source_rows: list[tuple[int, list[str]]],
    split: str,
    label_to_index: dict[str, int] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, source_row in source_rows:
        text, label = validate_row(source_row, split, source_index)
        row: dict[str, Any] = {
            "row_id": f"{split}|{source_index:05d}",
            "source_index": source_index,
            "split": split,
            "text": text,
        }
        if label_to_index is not None:
            if label not in label_to_index:
                raise ValueError(f"unknown intent label {label!r} in {split}")
            row["intent"] = label
            row["target_index"] = label_to_index[label]
        else:
            if label != "oos":
                raise ValueError(f"expected OOS label in {split}, found {label!r}")
        rows.append(row)
    return rows


def validate_population(
    source: dict[str, list[list[str]]],
    intents: list[str],
) -> None:
    constants = SCIENTIFIC_CONSTANTS
    expected = {
        "train": constants["intent_count"] * constants["train_per_intent"],
        "val": constants["intent_count"] * constants["validation_per_intent"],
        "test": constants["intent_count"] * constants["test_per_intent"],
        "oos_train": 100,
        "oos_val": constants["validation_oos_count"],
        "oos_test": constants["test_oos_count"],
    }
    for split, count in expected.items():
        if len(source[split]) != count:
            raise ValueError(f"{split} count {len(source[split])} != {count}")
    if len(intents) != constants["intent_count"]:
        raise ValueError(f"intent count {len(intents)} != {constants['intent_count']}")

    for split, per_intent in (
        ("train", constants["train_per_intent"]),
        ("val", constants["validation_per_intent"]),
        ("test", constants["test_per_intent"]),
    ):
        counts = {intent: 0 for intent in intents}
        for index, row in enumerate(source[split]):
            _, label = validate_row(row, split, index)
            if label not in counts:
                raise ValueError(f"unexpected {split} label {label!r}")
            counts[label] += 1
        bad = {label: count for label, count in counts.items() if count != per_intent}
        if bad:
            raise ValueError(f"unbalanced {split} labels: {bad}")

    for split, rows in source.items():
        split_texts: list[str] = []
        for index, row in enumerate(rows):
            text, _ = validate_row(row, split, index)
            split_texts.append(text)
        if len(set(split_texts)) != len(split_texts):
            raise ValueError(f"CLINC source contains duplicate texts within {split}")


def exclude_train_overlaps(
    source_rows: list[list[str]],
    split: str,
    train_texts: set[str],
) -> tuple[list[tuple[int, list[str]]], list[dict[str, Any]]]:
    """Remove exact train/evaluation text overlaps before any model access."""

    kept: list[tuple[int, list[str]]] = []
    excluded: list[dict[str, Any]] = []
    for source_index, source_row in enumerate(source_rows):
        text, label = validate_row(source_row, split, source_index)
        if text in train_texts:
            excluded.append(
                {
                    "source_index": source_index,
                    "split": split,
                    "text": text,
                    "label": label,
                    "reason": "exact text appears in training split",
                }
            )
        else:
            kept.append((source_index, source_row))
    return kept, excluded


def build_manifests(source_repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_repository = source_repository.resolve()
    source_path = source_repository / "data" / "data_full.json"
    license_path = source_repository / "LICENSE"
    domains_path = source_repository / "data" / "domains.json"
    if repository_head(source_repository) != SOURCE_COMMIT:
        raise ValueError("CLINC repository is not at the registered commit")
    if sha256_file(source_path) != SOURCE_DATA_SHA256:
        raise ValueError("CLINC data_full.json hash does not match the contract")

    source = load_source(source_path)
    intents = sorted({validate_row(row, "train", index)[1] for index, row in enumerate(source["train"])})
    validate_population(source, intents)
    label_to_index = {label: index for index, label in enumerate(intents)}
    train_texts = {
        validate_row(row, "train", index)[0]
        for index, row in enumerate(source["train"])
    }
    validation_rows, validation_exclusions = exclude_train_overlaps(
        source["val"], "val", train_texts
    )
    test_rows, test_exclusions = exclude_train_overlaps(
        source["test"], "test", train_texts
    )
    if len(validation_exclusions) != SCIENTIFIC_CONSTANTS["validation_train_overlap_exclusions"]:
        raise ValueError("validation/train overlap count differs from the blind census")
    if len(test_exclusions) != SCIENTIFIC_CONSTANTS["test_train_overlap_exclusions"]:
        raise ValueError("test/train overlap count differs from the blind census")
    if len(validation_rows) != SCIENTIFIC_CONSTANTS["validation_in_scope_count"]:
        raise ValueError("filtered validation count differs from the contract")
    if len(test_rows) != SCIENTIFIC_CONSTANTS["test_in_scope_count"]:
        raise ValueError("filtered test count differs from the contract")

    source_record = {
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "data_path": "data/data_full.json",
        "data_sha256": sha256_file(source_path),
        "domains_sha256": sha256_file(domains_path),
        "license_path": "LICENSE",
        "license_sha256": sha256_file(license_path),
        "license_spdx_note": "CC-BY-3.0",
    }
    construction = {
        "label_order": "lexicographic",
        "row_order": "official source order",
        "prototype_blocks": "four contiguous 25-row blocks per intent",
        "evaluation_overlap_rule": "exclude exact text present in training",
        "encoder_loaded": False,
        "routing_outcomes_computed": False,
    }

    development = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "phase": "development",
        "source": source_record,
        "construction": construction,
        "counts": {
            "intents": len(intents),
            "train": len(source["train"]),
            "in_scope": len(validation_rows),
            "oos": len(source["oos_val"]),
            "excluded_train_overlaps": len(validation_exclusions),
        },
        "intents": intents,
        "exclusions": validation_exclusions,
        "train": make_rows(list(enumerate(source["train"])), "train", label_to_index),
        "in_scope": make_rows(validation_rows, "val", label_to_index),
        "oos": make_rows(list(enumerate(source["oos_val"])), "oos_val", None),
    }
    test = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "phase": "test",
        "source": source_record,
        "construction": construction,
        "counts": {
            "intents": len(intents),
            "in_scope": len(test_rows),
            "oos": len(source["oos_test"]),
            "excluded_train_overlaps": len(test_exclusions),
        },
        "intents": intents,
        "exclusions": test_exclusions,
        "in_scope": make_rows(test_rows, "test", label_to_index),
        "oos": make_rows(list(enumerate(source["oos_test"])), "oos_test", None),
    }
    return development, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    development, test = build_manifests(args.source_repo)
    write_json(DEVELOPMENT_MANIFEST_PATH, development)
    write_json(TEST_MANIFEST_PATH, test)
    print(f"wrote {DEVELOPMENT_MANIFEST_PATH}")
    print(f"wrote {TEST_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
