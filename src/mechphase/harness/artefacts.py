"""Writers for the root artefacts and the integrity manifest.

Ref: Data availability statement (the provenance file and the pipeline constants).

Every writer takes an explicit root so that a test can direct the output to a temporary
directory and never overwrite the shipped artefacts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mechphase import REPOSITORY_SLUG
from mechphase.support.io import (
    iter_files,
    read_json,
    sha256_file,
    sha256_mapping,
    write_json,
    write_text,
)
from mechphase.support.types import CheckStatus, OverallVerdict

SKIP_DIRECTORIES: tuple[str, ...] = (
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    "runs",
)
MANIFEST_NAME = "integrity_manifest.json"
SUMMARY_NAME = "verification_summary.txt"
CLAIM_NAME = "claim_to_code.json"
REPORT_NAME = "verification_report.json"
DATASET_NAME = "dataset_urls.txt"


def write_claim_to_code(root: Path, payload: Mapping[str, Any]) -> Path:
    path = root / CLAIM_NAME
    write_json(path, payload)
    return path


def write_verification_report(root: Path, payload: Mapping[str, Any]) -> Path:
    path = root / REPORT_NAME
    write_json(path, payload)
    return path


def write_summary(root: Path, text: str) -> Path:
    path = root / SUMMARY_NAME
    write_text(path, text)
    return path


def write_dataset_urls(root: Path, entries: Sequence[str]) -> Path:
    path = root / DATASET_NAME
    write_text(path, "\n".join(entries))
    return path



def write_integrity_manifest(
    root: Path,
    exclude: Iterable[str] = (MANIFEST_NAME,),
    skip_directories: Sequence[str] = SKIP_DIRECTORIES,
) -> dict[str, Any]:
    blocked = set(exclude)
    digests: dict[str, str] = {}
    for path in iter_files(root, skip_dirs=skip_directories):
        relative = path.relative_to(root).as_posix()
        if relative in blocked or path.name.startswith(".tmp-") or path.name.startswith(".ckpt-"):
            continue
        digests[relative] = sha256_file(path)
    payload: dict[str, Any] = {
        "root": REPOSITORY_SLUG,
        "file_count": len(digests),
        "files": digests,
        "manifest_digest": sha256_mapping(digests),
    }
    write_json(root / MANIFEST_NAME, payload)
    return payload


def verify_manifest(
    root: Path,
    payload: Mapping[str, Any] | None = None,
    skip_directories: Sequence[str] = SKIP_DIRECTORIES,
) -> dict[str, Any]:
    recorded = dict(payload) if payload is not None else dict(read_json(root / MANIFEST_NAME))
    files = dict(recorded.get("files", {}))
    mismatched: list[str] = []
    missing: list[str] = []
    for relative, digest in files.items():
        path = root / relative
        if not path.exists():
            missing.append(relative)
            continue
        if sha256_file(path) != digest:
            mismatched.append(relative)
    live = {
        path.relative_to(root).as_posix()
        for path in iter_files(root, skip_dirs=skip_directories)
        if path.name != MANIFEST_NAME
    }
    untracked = sorted(live - set(files))
    return {
        "recorded_files": len(files),
        "missing": sorted(missing),
        "mismatched": sorted(mismatched),
        "untracked": untracked,
        "consistent": not missing and not mismatched and not untracked,
    }

def status_counts(checks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status.value: 0 for status in CheckStatus}
    for check in checks:
        counts[str(check["status"])] = counts.get(str(check["status"]), 0) + 1
    return counts


def overall_verdict(checks: Sequence[Mapping[str, Any]]) -> str:
    counts = status_counts(checks)
    if counts.get(CheckStatus.FAIL.value, 0) > 0:
        return OverallVerdict.UNVERIFIED.value
    outstanding = counts.get(CheckStatus.NOT_RUN.value, 0) + counts.get(CheckStatus.BLOCKED.value, 0)
    if outstanding > 0:
        return OverallVerdict.PARTIALLY_VERIFIED.value
    return OverallVerdict.VERIFIED.value


def render_check_table(checks: Sequence[Mapping[str, Any]], width: int = 36) -> str:
    lines = [f"{'name'.ljust(width)} status", "-" * (width + 7)]
    for check in checks:
        lines.append(f"{str(check['name']).ljust(width)} {check['status']}")
    return "\n".join(lines)



def render_mapping_table(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str], widths: Sequence[int]
) -> str:
    header = "  ".join(key.ljust(width) for key, width in zip(keys, widths, strict=True))
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            "  ".join(
                str(row.get(key, ""))[:width].ljust(width)
                for key, width in zip(keys, widths, strict=True)
            )
        )
    return "\n".join(lines)

