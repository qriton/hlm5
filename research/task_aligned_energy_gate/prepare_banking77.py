"""Build outcome-blind Banking77 development and sealed-test manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from contract import (
    DEVELOPMENT_MANIFEST_PATH,
    EXPERIMENT_ID,
    SCIENTIFIC_CONSTANTS,
    SOURCE_COMMIT,
    SOURCE_HASHES,
    SOURCE_REPOSITORY,
    SPLIT_SALT,
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


def normalize_text(text: str) -> str:
    """Return the registered non-semantic duplicate key."""

    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return " ".join(normalized.split())


def load_csv(path: Path, split: str, categories: list[str]) -> list[dict[str, Any]]:
    category_set = set(categories)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"unexpected {split} CSV header: {reader.fieldnames}")
        for source_index, row in enumerate(reader):
            text = row.get("text")
            category = row.get("category")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{split}[{source_index}] has invalid text")
            if category not in category_set:
                raise ValueError(f"{split}[{source_index}] has unknown category")
            rows.append(
                {
                    "row_id": f"{split}|{source_index:05d}",
                    "source_index": source_index,
                    "split": split,
                    "text": text,
                    "normalized_text_sha256": hashlib.sha256(
                        normalize_text(text).encode("utf-8")
                    ).hexdigest(),
                    "intent": category,
                    "target_index": categories.index(category),
                }
            )
    return rows


def split_key(row: dict[str, Any]) -> str:
    normalized = normalize_text(str(row["text"]))
    material = f"{SPLIT_SALT}\0{row['intent']}\0{normalized}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def deduplicate_training(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_text(str(row["text"]))].append(row)

    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for normalized, group in grouped.items():
        labels = {str(row["intent"]) for row in group}
        if len(labels) != 1:
            raise ValueError(f"normalized training label conflict: {normalized!r}")
        ordered = sorted(group, key=lambda row: int(row["source_index"]))
        kept.append(ordered[0])
        for row in ordered[1:]:
            excluded.append(
                {
                    **row,
                    "reason": "later source row duplicates normalized training text",
                    "kept_row_id": ordered[0]["row_id"],
                }
            )
    return sorted(kept, key=lambda row: int(row["source_index"])), sorted(
        excluded, key=lambda row: int(row["source_index"])
    )


def assign_training_roles(
    rows: list[dict[str, Any]], categories: list[str]
) -> dict[str, list[dict[str, Any]]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["intent"])].append(row)

    calibration_per = int(SCIENTIFIC_CONSTANTS["calibration_per_intent"])
    development_per = int(SCIENTIFIC_CONSTANTS["development_per_intent"])
    roles: dict[str, list[dict[str, Any]]] = {
        "fit": [],
        "calibration": [],
        "development": [],
    }
    for category in categories:
        ordered = sorted(
            by_category[category],
            key=lambda row: (split_key(row), int(row["source_index"])),
        )
        if len(ordered) <= calibration_per + development_per:
            raise ValueError(f"too few unique rows for {category}")
        calibration = ordered[:calibration_per]
        development = ordered[calibration_per : calibration_per + development_per]
        fit = ordered[calibration_per + development_per :]
        for role, role_rows in (
            ("fit", fit),
            ("calibration", calibration),
            ("development", development),
        ):
            for row in role_rows:
                roles[role].append({**row, "role": role, "split_key": split_key(row)})

    for role in roles:
        roles[role].sort(key=lambda row: (int(row["target_index"]), row["split_key"]))
    return roles


def partition_test(
    rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_by_normalized: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        train_by_normalized[normalize_text(str(row["text"]))] = row

    primary: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        normalized = normalize_text(str(row["text"]))
        train_row = train_by_normalized.get(normalized)
        if train_row is None:
            primary.append(row)
            continue
        if train_row["intent"] != row["intent"]:
            raise ValueError("normalized train/test overlap has conflicting labels")
        excluded.append(
            {
                **row,
                "reason": "normalized text appears in official training",
                "train_row_id": train_row["row_id"],
            }
        )
    return primary, excluded


def validate_counts(
    official_train: list[dict[str, Any]],
    unique_train: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    roles: dict[str, list[dict[str, Any]]],
    official_test: list[dict[str, Any]],
    primary_test: list[dict[str, Any]],
    test_exclusions: list[dict[str, Any]],
    categories: list[str],
) -> None:
    constants = SCIENTIFIC_CONSTANTS
    expected = {
        "official train": (len(official_train), constants["official_train_count"]),
        "unique train": (len(unique_train), constants["deduplicated_train_count"]),
        "fit": (len(roles["fit"]), constants["fit_count"]),
        "calibration": (len(roles["calibration"]), constants["calibration_count"]),
        "development": (len(roles["development"]), constants["development_count"]),
        "official test": (len(official_test), constants["official_test_count"]),
        "primary test": (len(primary_test), constants["primary_test_count"]),
        "test exclusions": (
            len(test_exclusions),
            constants["normalized_train_test_overlap_count"],
        ),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"{name} count {actual} != {wanted}")
    if len(exclusions) != constants["normalized_train_duplicate_groups"]:
        raise ValueError("normalized training duplicate count differs from census")
    if len(categories) != constants["intent_count"] or len(set(categories)) != len(
        categories
    ):
        raise ValueError("category list is not exactly 77 unique labels")

    for role in ("calibration", "development"):
        counts = defaultdict(int)
        for row in roles[role]:
            counts[str(row["intent"])] += 1
        expected_per = int(constants[f"{role}_per_intent"])
        if set(counts) != set(categories) or set(counts.values()) != {expected_per}:
            raise ValueError(f"{role} is not exactly balanced")

    role_ids = [{str(row["row_id"]) for row in roles[role]} for role in roles]
    if any(role_ids[i] & role_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("training roles overlap")


def build_manifests(source_repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_repository = source_repository.resolve()
    if repository_head(source_repository) != SOURCE_COMMIT:
        raise ValueError("Banking77 repository is not at the registered commit")
    for relative, expected_hash in SOURCE_HASHES.items():
        path = source_repository / relative
        if sha256_file(path) != expected_hash:
            raise ValueError(f"source hash mismatch: {relative}")

    categories_value = json.loads(
        (source_repository / "banking_data" / "categories.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(categories_value, list) or not all(
        isinstance(value, str) for value in categories_value
    ):
        raise ValueError("categories.json must be a string list")
    categories = list(categories_value)
    official_train = load_csv(
        source_repository / "banking_data" / "train.csv", "train", categories
    )
    official_test = load_csv(
        source_repository / "banking_data" / "test.csv", "test", categories
    )
    unique_train, train_exclusions = deduplicate_training(official_train)
    roles = assign_training_roles(unique_train, categories)
    primary_test, test_exclusions = partition_test(official_test, official_train)
    validate_counts(
        official_train,
        unique_train,
        train_exclusions,
        roles,
        official_test,
        primary_test,
        test_exclusions,
        categories,
    )

    source_record = {
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "hashes": SOURCE_HASHES,
        "license_spdx_note": "CC-BY-4.0",
    }
    construction = {
        "normalization": "Unicode NFKC, casefold, strip, collapse whitespace",
        "deduplication": "keep lowest source index per normalized training text",
        "role_order": "sha256(split_salt NUL intent NUL normalized_text), source_index",
        "role_assignment": "first 10 calibration, next 10 development, remainder fit",
        "test_primary_rule": "exclude normalized text present in official training",
        "encoder_loaded": False,
        "model_outcomes_computed": False,
    }
    development = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "phase": "development",
        "source": source_record,
        "construction": construction,
        "categories": categories,
        "counts": {
            "official_train": len(official_train),
            "deduplicated_train": len(unique_train),
            "fit": len(roles["fit"]),
            "calibration": len(roles["calibration"]),
            "development": len(roles["development"]),
            "train_duplicate_exclusions": len(train_exclusions),
        },
        "train_duplicate_exclusions": train_exclusions,
        **roles,
    }
    test = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "phase": "test",
        "source": source_record,
        "construction": construction,
        "categories": categories,
        "counts": {
            "official": len(official_test),
            "primary": len(primary_test),
            "normalized_train_overlap_exclusions": len(test_exclusions),
        },
        "primary": primary_test,
        "official": official_test,
        "normalized_train_overlap_exclusions": test_exclusions,
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
