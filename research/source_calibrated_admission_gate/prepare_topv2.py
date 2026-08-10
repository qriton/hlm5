"""Build the outcome-blind TOPv2 populations for the registered SCA-1 assay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .contract import (
    DEFAULT_DATA_ROOT,
    GROUP_INTENT_COUNT,
    INTENT_ORDER_SALT,
    MANIFEST_PATH,
    PARTITIONS,
    ROW_ORDER_SALT,
    SOURCE_DOMAINS,
    SOURCE_FIT_PER_DOMAIN,
    SOURCE_FIT_SALT,
    SUPPORT_PER_INTENT,
    TARGET_DOMAINS,
    ROWS_PER_INTENT,
    sha256_bytes,
    stable_sha256,
    verify_data_files,
    write_json,
)


ROOT_INTENT_RE = re.compile(r"^\[IN:([A-Z0-9_]+)(?:\s|\])")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("empty normalized utterance")
    return normalized


def parse_root_intent(semantic_parse: str) -> str:
    match = ROOT_INTENT_RE.match(semantic_parse.strip())
    if match is None:
        raise ValueError(f"semantic parse has no root intent: {semantic_parse!r}")
    return match.group(1)


def _row_id(
    domain: str,
    partition: str,
    source_line: int,
    normalized: str,
    intent: str,
    semantic_parse: str,
) -> str:
    payload = "\0".join(
        (
            domain,
            partition,
            str(source_line),
            normalized,
            intent,
            semantic_parse,
        )
    )
    return sha256_bytes(payload.encode("utf-8"))


def read_split(
    path: Path, expected_domain: str, partition: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["domain", "utterance", "semantic_parse"]:
            raise ValueError(f"unexpected TOPv2 header in {path}: {reader.fieldnames}")
        for source_line, source in enumerate(reader, start=2):
            if source["domain"] != expected_domain:
                raise ValueError(
                    f"domain mismatch in {path}:{source_line}: {source['domain']}"
                )
            utterance = source["utterance"]
            semantic_parse = source["semantic_parse"]
            normalized = normalize_text(utterance)
            root_intent = parse_root_intent(semantic_parse)
            intent = f"{expected_domain}::{root_intent}"
            rows.append(
                {
                    "row_id": _row_id(
                        expected_domain,
                        partition,
                        source_line,
                        normalized,
                        intent,
                        semantic_parse,
                    ),
                    "domain": expected_domain,
                    "partition": partition,
                    "source_line": source_line,
                    "intent": intent,
                    "utterance": utterance,
                    "normalized": normalized,
                    "utterance_sha256": sha256_bytes(normalized.encode("utf-8")),
                }
            )
    return rows


def read_all_rows(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {partition: [] for partition in PARTITIONS}
    for domain in SOURCE_DOMAINS + TARGET_DOMAINS:
        for partition in PARTITIONS:
            rows[partition].extend(
                read_split(
                    data_root / f"{domain}_{partition}.tsv",
                    domain,
                    partition,
                )
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
    blocked_rows = 0
    for normalized in sorted(grouped):
        group = grouped[normalized]
        if normalized in blocked_normalized:
            blocked_rows += len(group)
            continue
        labels = {row["intent"] for row in group}
        if len(labels) != 1:
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
        "overlap_rows_dropped": blocked_rows,
    }


def row_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row["row_id"],
        "domain": row["domain"],
        "partition": row["partition"],
        "source_line": row["source_line"],
        "intent": row["intent"],
        "utterance_sha256": row["utterance_sha256"],
    }


def population_sha256(rows: Iterable[dict[str, Any]]) -> str:
    return stable_sha256([row_descriptor(row) for row in rows])


def _salted_rank(value: str, salt: str) -> tuple[str, str]:
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest(), value


def select_rows(
    rows: Iterable[dict[str, Any]], count: int, salt: str
) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: _salted_rank(row["row_id"], salt))
    if len(rows) < count:
        raise ValueError(f"requested {count} rows from a population of {len(rows)}")
    return rows[:count]


def intent_counts(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    return Counter(row["intent"] for row in rows)


def eligible_intents(
    train_rows: Iterable[dict[str, Any]],
    eval_rows: Iterable[dict[str, Any]],
    domains: tuple[str, ...],
) -> list[str]:
    train = intent_counts(row for row in train_rows if row["domain"] in domains)
    evaluation = intent_counts(row for row in eval_rows if row["domain"] in domains)
    return sorted(
        intent
        for intent, count in train.items()
        if count >= SUPPORT_PER_INTENT and evaluation[intent] >= ROWS_PER_INTENT
    )


def assign_source_intent_groups(
    intents: Iterable[str],
    group_size: int = GROUP_INTENT_COUNT,
) -> dict[str, list[str]]:
    ordered = sorted(
        intents, key=lambda intent: _salted_rank(intent, INTENT_ORDER_SALT)
    )
    needed = 3 * group_size
    if len(ordered) < needed:
        raise ValueError(f"need {needed} eligible source intents, found {len(ordered)}")
    return {
        "calibration_supported": ordered[:group_size],
        "calibration_negative": ordered[group_size : 2 * group_size],
        "off_support_audit": ordered[2 * group_size : 3 * group_size],
        "unused": ordered[3 * group_size :],
    }


def _rows_for_intents(
    rows: Iterable[dict[str, Any]],
    intents: Iterable[str],
    count_per_intent: int,
    salt_prefix: str,
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
                f"{salt_prefix}\0{intent}",
            )
        )
    return selected


def _population_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "sha256": population_sha256(rows),
    }


def build_populations(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    all_rows = read_all_rows(data_root)
    raw_train_norms = {row["normalized"] for row in all_rows["train"]}
    raw_eval_norms = {row["normalized"] for row in all_rows["eval"]}

    train_rows, train_cleanup = clean_partition(all_rows["train"])
    eval_rows, eval_cleanup = clean_partition(all_rows["eval"], raw_train_norms)
    test_rows, test_cleanup = clean_partition(
        all_rows["test"], raw_train_norms | raw_eval_norms
    )

    target_intents = eligible_intents(train_rows, eval_rows, tuple(TARGET_DOMAINS))
    if len(target_intents) != GROUP_INTENT_COUNT:
        raise ValueError(
            f"registered target intent count changed: {len(target_intents)}"
        )
    source_intents = eligible_intents(train_rows, eval_rows, tuple(SOURCE_DOMAINS))
    source_groups = assign_source_intent_groups(source_intents)

    source_fit: list[dict[str, Any]] = []
    for domain in SOURCE_DOMAINS:
        source_fit.extend(
            select_rows(
                (row for row in train_rows if row["domain"] == domain),
                SOURCE_FIT_PER_DOMAIN,
                f"{SOURCE_FIT_SALT}\0{domain}",
            )
        )

    target_support = _rows_for_intents(
        train_rows,
        target_intents,
        SUPPORT_PER_INTENT,
        f"{ROW_ORDER_SALT}\0target-support",
    )
    target_development = _rows_for_intents(
        eval_rows,
        target_intents,
        ROWS_PER_INTENT,
        f"{ROW_ORDER_SALT}\0target-development",
    )
    calibration_support = _rows_for_intents(
        train_rows,
        source_groups["calibration_supported"],
        SUPPORT_PER_INTENT,
        f"{ROW_ORDER_SALT}\0calibration-support",
    )
    calibration_positive = _rows_for_intents(
        eval_rows,
        source_groups["calibration_supported"],
        ROWS_PER_INTENT,
        f"{ROW_ORDER_SALT}\0calibration-positive",
    )
    calibration_negative = _rows_for_intents(
        eval_rows,
        source_groups["calibration_negative"],
        ROWS_PER_INTENT,
        f"{ROW_ORDER_SALT}\0calibration-negative",
    )
    off_support_audit = _rows_for_intents(
        eval_rows,
        source_groups["off_support_audit"],
        ROWS_PER_INTENT,
        f"{ROW_ORDER_SALT}\0off-support-audit",
    )
    test_target = _rows_for_intents(
        test_rows,
        target_intents,
        ROWS_PER_INTENT,
        f"{ROW_ORDER_SALT}\0test-target",
    )

    hidden_population_names = (
        "source_fit",
        "target_support",
        "target_development",
        "calibration_support",
        "calibration_positive",
        "calibration_negative",
        "off_support_audit",
    )
    populations = {
        "source_fit": source_fit,
        "target_support": target_support,
        "target_development": target_development,
        "calibration_support": calibration_support,
        "calibration_positive": calibration_positive,
        "calibration_negative": calibration_negative,
        "off_support_audit": off_support_audit,
        "test_target": test_target,
    }
    hidden_ids = {
        row["row_id"] for name in hidden_population_names for row in populations[name]
    }
    test_ids = {row["row_id"] for row in test_target}
    if hidden_ids & test_ids:
        raise AssertionError("test row entered the development hidden population")
    if any(
        row["partition"] == "test"
        for name in hidden_population_names
        for row in populations[name]
    ):
        raise AssertionError("test partition entered a development population")

    return {
        "rows": populations,
        "metadata": {
            "cleanup": {
                "train": train_cleanup,
                "eval": eval_cleanup,
                "test": test_cleanup,
            },
            "raw_counts": {
                partition: {
                    domain: sum(row["domain"] == domain for row in all_rows[partition])
                    for domain in SOURCE_DOMAINS + TARGET_DOMAINS
                }
                for partition in PARTITIONS
            },
            "target_intents": target_intents,
            "source_eligible_intents": source_intents,
            "source_groups": source_groups,
            "test_utterances_tokenized_or_encoded": False,
        },
    }


def make_manifest(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    file_hashes = verify_data_files(data_root)
    built = build_populations(data_root)
    rows = built["rows"]
    manifest = {
        "schema": "hlm5-sca1-topv2-development-manifest-v1",
        "source": {
            "name": "TOPv2 1.1",
            "root": str(data_root),
            "official_url": "https://dl.fbaipublicfiles.com/topv2/TOPv2_Dataset.zip",
            "archive_sha256": "e73a13eb32f67f69c630abde549fa153f288564e2703b71492f04cb59b2cae13",
            "files": file_hashes,
            "license": "CC BY-SA 4.0",
        },
        "constants": {
            "source_domains": list(SOURCE_DOMAINS),
            "target_domains": list(TARGET_DOMAINS),
            "support_per_intent": SUPPORT_PER_INTENT,
            "rows_per_intent": ROWS_PER_INTENT,
            "source_fit_per_domain": SOURCE_FIT_PER_DOMAIN,
            "group_intent_count": GROUP_INTENT_COUNT,
            "intent_order_salt": INTENT_ORDER_SALT,
            "row_order_salt": ROW_ORDER_SALT,
            "source_fit_salt": SOURCE_FIT_SALT,
        },
        "metadata": built["metadata"],
        "populations": {
            name: _population_record(value) for name, value in rows.items()
        },
        "support_rows": {
            name: [row_descriptor(row) for row in rows[name]]
            for name in ("calibration_support", "target_support")
        },
        "development_hidden_population_names": [
            "source_fit",
            "target_support",
            "target_development",
            "calibration_support",
            "calibration_positive",
            "calibration_negative",
            "off_support_audit",
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
    import json

    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("TOPv2 development manifest does not match source files")
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
