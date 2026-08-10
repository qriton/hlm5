"""Build the outcome-blind, deterministic ParaRel manifest for HLM5 S1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from contract import (
    MANIFEST_PATH,
    PARAREL_COMMIT,
    PARAREL_REPOSITORY,
    REGISTRATION_PATH,
    SCIENTIFIC_CONSTANTS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected an object in {path}:{line_number}")
        rows.append(value)
    return rows


def render(pattern: str, subject: str, mask_literal: str) -> str:
    if pattern.count("[X]") != 1 or pattern.count("[Y]") != 1:
        raise ValueError(f"pattern must contain one [X] and one [Y]: {pattern!r}")
    rendered = pattern.replace("[X]", subject).replace("[Y]", mask_literal)
    if "[X]" in rendered or "[Y]" in rendered:
        raise ValueError(f"unresolved placeholder in {rendered!r}")
    return " ".join(rendered.split())


def selected_unique_facts(
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        try:
            normalized.append(
                {
                    "uuid": str(row["uuid"]),
                    "sub_label": str(row["sub_label"]),
                    "obj_label": str(row["obj_label"]),
                }
            )
        except KeyError as exc:
            raise ValueError(f"fact row is missing {exc.args[0]}") from exc
    normalized.sort(key=lambda row: (row["uuid"], row["sub_label"], row["obj_label"]))
    selected: list[dict[str, str]] = []
    subjects: set[str] = set()
    for row in normalized:
        if row["sub_label"] in subjects:
            continue
        selected.append(row)
        subjects.add(row["sub_label"])
        if len(selected) == count:
            return selected
    raise ValueError(f"only {len(selected)} unique subjects; required {count}")


def source_commit(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_manifest(
    source_root: Path,
    *,
    facts_per_relation: int,
    mask_literal: str,
) -> dict[str, Any]:
    pattern_dir = source_root / "data" / "pattern_data" / "graphs_json"
    fact_dir = source_root / "data" / "trex_lms_vocab"
    if not pattern_dir.is_dir() or not fact_dir.is_dir():
        raise FileNotFoundError("ParaRel pattern or fact directory is missing")

    relation_inputs: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    repository_pattern_count = 0
    for pattern_path in sorted(pattern_dir.glob("*.jsonl"), key=lambda path: path.name):
        relation = pattern_path.stem
        fact_path = fact_dir / pattern_path.name
        if not fact_path.is_file():
            raise FileNotFoundError(f"missing fact file for {relation}")
        patterns = load_jsonl(pattern_path)
        facts = load_jsonl(fact_path)
        repository_pattern_count += len(patterns)
        for path in (pattern_path, fact_path):
            relative = path.relative_to(source_root).as_posix()
            source_hashes[relative] = sha256_file(path)
        if len(patterns) < 2:
            continue
        pattern_strings = [str(row["pattern"]) for row in patterns]
        relation_inputs.append(
            {
                "relation": relation,
                "patterns": pattern_strings,
                "facts": selected_unique_facts(facts, facts_per_relation),
                "source_fact_count": len(facts),
            }
        )

    relation_inputs.sort(key=lambda item: item["relation"])
    slots: list[dict[str, Any]] = []
    slot_by_pair: dict[tuple[str, str], int] = {}
    for relation_data in relation_inputs:
        relation = relation_data["relation"]
        canonical_pattern = relation_data["patterns"][0]
        for fact in relation_data["facts"]:
            slot = len(slots)
            pair = (relation, fact["sub_label"])
            if pair in slot_by_pair:
                raise ValueError(f"duplicate stored address {pair}")
            slot_by_pair[pair] = slot
            slots.append(
                {
                    "slot": slot,
                    "relation": relation,
                    "uuid": fact["uuid"],
                    "subject": fact["sub_label"],
                    "object": fact["obj_label"],
                    "canonical_pattern": canonical_pattern,
                    "text": render(canonical_pattern, fact["sub_label"], mask_literal),
                }
            )

    relation_index = {
        relation_data["relation"]: index
        for index, relation_data in enumerate(relation_inputs)
    }
    queries: list[dict[str, Any]] = []
    for relation_data in relation_inputs:
        relation = relation_data["relation"]
        current_index = relation_index[relation]
        for fact_index, fact in enumerate(relation_data["facts"]):
            target_slot = slot_by_pair[(relation, fact["sub_label"])]
            off_subject: str | None = None
            for offset in range(1, len(relation_inputs) + 1):
                other = relation_inputs[(current_index + offset) % len(relation_inputs)]
                candidate = other["facts"][fact_index % len(other["facts"])]["sub_label"]
                if (relation, candidate) not in slot_by_pair:
                    off_subject = candidate
                    break
            if off_subject is None:
                raise ValueError(f"could not construct off-support subject for {relation}")

            for pattern_index, pattern in enumerate(relation_data["patterns"][1:], 1):
                query_id = f"{relation}|{fact['uuid']}|pattern-{pattern_index:02d}"
                queries.append(
                    {
                        "query_id": query_id,
                        "relation": relation,
                        "subject": fact["sub_label"],
                        "object": fact["obj_label"],
                        "target_slot": target_slot,
                        "pattern_index": pattern_index,
                        "pattern": pattern,
                        "text": render(pattern, fact["sub_label"], mask_literal),
                        "off_support_subject": off_subject,
                        "off_support_text": render(pattern, off_subject, mask_literal),
                    }
                )

    relation_summaries = [
        {
            "relation": item["relation"],
            "pattern_count": len(item["patterns"]),
            "source_fact_count": item["source_fact_count"],
            "selected_fact_count": len(item["facts"]),
        }
        for item in relation_inputs
    ]
    return {
        "schema_version": 1,
        "experiment_id": SCIENTIFIC_CONSTANTS["experiment_id"],
        "source": {
            "repository": PARAREL_REPOSITORY,
            "commit": source_commit(source_root),
            "file_sha256": dict(sorted(source_hashes.items())),
            "repository_relation_count": len(list(pattern_dir.glob("*.jsonl"))),
            "repository_pattern_count": repository_pattern_count,
        },
        "construction": {
            "facts_per_relation": facts_per_relation,
            "mask_literal": mask_literal,
            "canonical_pattern_index": 0,
            "fact_sort": ["uuid", "sub_label", "obj_label"],
        },
        "counts": {
            "eligible_relations": len(relation_inputs),
            "eligible_patterns": sum(item["pattern_count"] for item in relation_summaries),
            "stored_addresses": len(slots),
            "supported_queries": len(queries),
            "off_support_queries": len(queries),
        },
        "relations": relation_summaries,
        "slots": slots,
        "queries": queries,
    }


def validate_registered_counts(manifest: dict[str, Any]) -> None:
    expected = {
        "eligible_relations": SCIENTIFIC_CONSTANTS["eligible_relation_count"],
        "eligible_patterns": SCIENTIFIC_CONSTANTS["pattern_count"],
        "stored_addresses": SCIENTIFIC_CONSTANTS["stored_address_count"],
        "supported_queries": SCIENTIFIC_CONSTANTS["supported_query_count"],
        "off_support_queries": SCIENTIFIC_CONSTANTS["off_support_query_count"],
    }
    if manifest["source"]["commit"] != PARAREL_COMMIT:
        raise RuntimeError(
            f"ParaRel commit mismatch: {manifest['source']['commit']} != {PARAREL_COMMIT}"
        )
    if manifest["counts"] != expected:
        raise RuntimeError(f"population mismatch: {manifest['counts']} != {expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="ParaRel checkout pinned to the registered commit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if REGISTRATION_PATH.exists():
        raise RuntimeError("registration exists; manifest regeneration is locked")
    manifest = build_manifest(
        args.source_root.resolve(),
        facts_per_relation=int(SCIENTIFIC_CONSTANTS["facts_per_relation"]),
        mask_literal=str(SCIENTIFIC_CONSTANTS["mask_literal"]),
    )
    validate_registered_counts(manifest)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote outcome-blind manifest: {MANIFEST_PATH}")
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
