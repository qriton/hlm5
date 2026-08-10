"""Outcome-blind MASSIVE population construction for NKT-1."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .contract import (
    DEFAULT_DATA_PATH,
    MANIFEST_PATH,
    SHUFFLE_SALT,
    SUPPORT_PER_INTENT,
    SUPPORT_SALT,
    TARGET_DOMAIN_COUNT,
    TARGET_DOMAIN_SALT,
    stable_sha256,
    verify_expected_file,
    write_json,
)


PARTITIONS = ("train", "dev", "test")


def normalize_text(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", text).casefold().strip(),
    )


def read_source(path: Path) -> list[dict[str, Any]]:
    verify_expected_file(path, "data")
    rows: list[dict[str, Any]] = []
    for source_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        row = {
            "id": str(raw["id"]),
            "partition": str(raw["partition"]),
            "scenario": str(raw["scenario"]),
            "intent": str(raw["intent"]),
            "utterance": str(raw["utt"]),
            "normalized_text": normalize_text(str(raw["utt"])),
            "source_index": source_index,
        }
        if row["partition"] not in PARTITIONS:
            raise ValueError(f"unexpected partition {row['partition']!r}")
        if not row["normalized_text"]:
            raise ValueError(f"empty normalized text at source index {source_index}")
        rows.append(row)
    return rows


def row_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "partition": row["partition"],
        "scenario": row["scenario"],
        "intent": row["intent"],
        "normalized_sha256": hashlib.sha256(
            row["normalized_text"].encode("utf-8")
        ).hexdigest(),
        "source_index": row["source_index"],
    }


def population_sha256(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: row["source_index"])
    return stable_sha256([row_descriptor(row) for row in ordered])


def clean_train(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row["partition"] == "train":
            groups[row["normalized_text"]].append(row)

    retained: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for group in groups.values():
        group = sorted(group, key=lambda row: row["source_index"])
        if len({row["intent"] for row in group}) > 1:
            conflict_rows.extend(group)
        else:
            retained.append(group[0])
            duplicate_rows.extend(group[1:])
    return (
        sorted(retained, key=lambda row: row["source_index"]),
        conflict_rows,
        duplicate_rows,
    )


def clean_evaluation(
    rows: list[dict[str, Any]],
    partition: str,
    blocked_normalized: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overlaps = [
        row
        for row in rows
        if row["partition"] == partition
        and row["normalized_text"] in blocked_normalized
    ]
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row["partition"] != partition:
            continue
        if row["normalized_text"] in blocked_normalized:
            continue
        groups[row["normalized_text"]].append(row)

    retained: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for group in groups.values():
        group = sorted(group, key=lambda row: row["source_index"])
        if len({row["intent"] for row in group}) != 1:
            raise ValueError(
                f"conflicting {partition} duplicate survived overlap cleanup"
            )
        retained.append(group[0])
        duplicates.extend(group[1:])
    return sorted(retained, key=lambda row: row["source_index"]), overlaps, duplicates


def rank_domains(domains: set[str]) -> list[tuple[str, str]]:
    ranked = []
    for domain in domains:
        digest = hashlib.sha256(
            f"{TARGET_DOMAIN_SALT}\0{domain}".encode("utf-8")
        ).hexdigest()
        ranked.append((digest, domain))
    return sorted(ranked)


def support_sort_key(row: dict[str, Any]) -> tuple[str, int]:
    payload = f"{SUPPORT_SALT}\0{row['intent']}\0{row['normalized_text']}\0{row['id']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), row["source_index"]


def shuffled_label_assignment(source_rows: list[dict[str, Any]]) -> list[str]:
    ordered = sorted(source_rows, key=lambda row: row["source_index"])
    labels = [row["intent"] for row in ordered]
    permutation = sorted(
        range(len(ordered)),
        key=lambda index: hashlib.sha256(
            (
                f"{SHUFFLE_SALT}\0{ordered[index]['id']}\0"
                f"{ordered[index]['source_index']}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    return [labels[index] for index in permutation]


def build_populations(path: Path) -> dict[str, Any]:
    rows = read_source(path)
    train, train_conflicts, train_duplicates = clean_train(rows)
    train_norm = {row["normalized_text"] for row in train}
    development, dev_overlaps, dev_duplicates = clean_evaluation(
        rows, "dev", train_norm
    )
    development_norm = {row["normalized_text"] for row in development}
    test, test_overlaps, test_duplicates = clean_evaluation(
        rows, "test", train_norm | development_norm
    )

    domains = {row["scenario"] for row in train}
    ranked_domains = rank_domains(domains)
    target_domains = [domain for _, domain in ranked_domains[:TARGET_DOMAIN_COUNT]]
    target_set = set(target_domains)
    source_domains = [domain for _, domain in ranked_domains[TARGET_DOMAIN_COUNT:]]

    source_train = [row for row in train if row["scenario"] not in target_set]
    target_train = [row for row in train if row["scenario"] in target_set]
    source_dev = [row for row in development if row["scenario"] not in target_set]
    target_dev = [row for row in development if row["scenario"] in target_set]
    source_test = [row for row in test if row["scenario"] not in target_set]
    target_test = [row for row in test if row["scenario"] in target_set]

    support_rows: list[dict[str, Any]] = []
    target_intents = sorted({row["intent"] for row in target_train})
    for intent in target_intents:
        candidates = sorted(
            (row for row in target_train if row["intent"] == intent),
            key=support_sort_key,
        )
        if len(candidates) < SUPPORT_PER_INTENT:
            raise ValueError(f"intent {intent!r} has fewer than three support rows")
        support_rows.extend(candidates[:SUPPORT_PER_INTENT])

    source_intents = sorted({row["intent"] for row in source_train})
    shuffled_labels = shuffled_label_assignment(source_train)
    if collections.Counter(shuffled_labels) != collections.Counter(
        row["intent"] for row in source_train
    ):
        raise AssertionError("shuffle did not preserve the source label multiset")

    return {
        "rows": rows,
        "train": train,
        "development": development,
        "test": test,
        "source_train": source_train,
        "target_train": target_train,
        "source_development": source_dev,
        "target_development": target_dev,
        "source_test": source_test,
        "target_test": target_test,
        "support_rows": support_rows,
        "source_intents": source_intents,
        "target_intents": target_intents,
        "target_domains": target_domains,
        "source_domains": source_domains,
        "ranked_domains": ranked_domains,
        "shuffled_labels": shuffled_labels,
        "exclusions": {
            "train_conflicts": train_conflicts,
            "train_duplicates": train_duplicates,
            "development_overlaps": dev_overlaps,
            "development_duplicates": dev_duplicates,
            "test_overlaps": test_overlaps,
            "test_duplicates": test_duplicates,
        },
    }


def make_manifest(path: Path) -> dict[str, Any]:
    populations = build_populations(path)
    exclusions = populations["exclusions"]
    target_test_intents = sorted({row["intent"] for row in populations["target_test"]})
    manifest = {
        "schema": "hlm5-nkt1-massive-population-v1",
        "source": {
            "dataset": "MASSIVE 1.1 en-US",
            "sha256": verify_expected_file(path, "data"),
            "row_count": len(populations["rows"]),
            "partition_counts": {
                partition: sum(
                    row["partition"] == partition for row in populations["rows"]
                )
                for partition in PARTITIONS
            },
            "license": "CC-BY-4.0",
        },
        "construction": {
            "normalization": "Unicode NFKC + casefold + trim + whitespace collapse",
            "target_domain_salt": TARGET_DOMAIN_SALT,
            "target_domain_count": TARGET_DOMAIN_COUNT,
            "support_salt": SUPPORT_SALT,
            "support_per_intent": SUPPORT_PER_INTENT,
            "shuffle_salt": SHUFFLE_SALT,
            "routing_outcomes_computed": False,
            "utterances_encoded": False,
            "test_utterances_encoded": False,
        },
        "domain_partition": {
            "ranked": [
                {"sha256": digest, "scenario": domain}
                for digest, domain in populations["ranked_domains"]
            ],
            "source_domains": populations["source_domains"],
            "target_domains": populations["target_domains"],
            "source_intents": populations["source_intents"],
            "target_intents": populations["target_intents"],
            "test_observed_target_intents": target_test_intents,
        },
        "counts": {
            "clean_train": len(populations["train"]),
            "source_train": len(populations["source_train"]),
            "target_train": len(populations["target_train"]),
            "primary_development": len(populations["development"]),
            "source_development_off_support": len(populations["source_development"]),
            "target_development": len(populations["target_development"]),
            "primary_test": len(populations["test"]),
            "source_test_off_support": len(populations["source_test"]),
            "target_test": len(populations["target_test"]),
            "support_slots": len(populations["support_rows"]),
            "source_intents": len(populations["source_intents"]),
            "target_intents": len(populations["target_intents"]),
            "test_observed_target_intents": len(target_test_intents),
            "excluded_train_conflict_rows": len(exclusions["train_conflicts"]),
            "excluded_train_duplicate_rows": len(exclusions["train_duplicates"]),
            "excluded_development_overlap_rows": len(
                exclusions["development_overlaps"]
            ),
            "excluded_development_duplicate_rows": len(
                exclusions["development_duplicates"]
            ),
            "excluded_test_overlap_rows": len(exclusions["test_overlaps"]),
            "excluded_test_duplicate_rows": len(exclusions["test_duplicates"]),
        },
        "population_sha256": {
            key: population_sha256(populations[key])
            for key in (
                "train",
                "source_train",
                "target_train",
                "development",
                "source_development",
                "target_development",
                "test",
                "source_test",
                "target_test",
                "support_rows",
            )
        },
        "support_rows": [row_descriptor(row) for row in populations["support_rows"]],
        "shuffled_source_labels_sha256": stable_sha256(populations["shuffled_labels"]),
    }
    return manifest


def verify_manifest(path: Path, manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = make_manifest(path)
    if observed != expected:
        raise RuntimeError("MASSIVE manifest does not match the current source")
    return observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.out.resolve()
    if output != MANIFEST_PATH.resolve():
        raise SystemExit(f"--out must be the registered path {MANIFEST_PATH}")
    manifest = make_manifest(args.source.resolve())
    write_json(output, manifest)
    print(f"wrote {output}")
    print(f"manifest_sha256={stable_sha256(manifest)}")
    print("routing_outcomes_computed=false")
    print("utterances_encoded=false")


if __name__ == "__main__":
    main()
