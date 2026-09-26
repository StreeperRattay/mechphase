"""Harness tests: configuration resolution, artefact writers and the audit driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mechphase.harness.artefacts import (
    CLAIM_NAME,
    DATASET_NAME,
    MANIFEST_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    overall_verdict,
    render_check_table,
    render_mapping_table,
    status_counts,
    verify_manifest,
    write_claim_to_code,
    write_dataset_urls,
    write_integrity_manifest,
    write_summary,
    write_verification_report,
)
from mechphase.harness.claims import (
    CLAIM_CHECKS,
    CLAIMS,
    claim_ids,
    claims_as_mapping,
    claims_for_module,
    code_symbols,
    statuses_from_checks,
)
from mechphase.harness.context import (
    build_context,
    framework_spec,
    objective_weights,
    synthetic_config,
    training_spec,
)
from mechphase.support.config import Configuration, repository_root, resolve_experiment
from mechphase.support.io import read_json, write_json
from mechphase.support.types import CheckStatus


def test_experiment_configurations_resolve() -> None:
    configuration = resolve_experiment("_smoke")
    assert configuration.as_str("name") == "_smoke"
    assert configuration.as_int("cohort.development_per_site") == 4
    assert configuration.as_int("training.epochs") == 3
    assert configuration.as_bool("framework.use_constant_phase_residual") is True
    main = resolve_experiment("main")
    assert main.as_int("training.world_size") == 4
    assert main.as_float("training.learning_rate") == pytest.approx(3.0e-5)
    assert main.as_str("training.precision") == "bf16"
    assert "provenance" in main
    overridden = resolve_experiment("_smoke", overrides=["training.epochs=7"])
    assert overridden.as_int("training.epochs") == 7
    with pytest.raises(FileNotFoundError):
        resolve_experiment("absent_experiment")


def test_configuration_helpers_build_specs() -> None:
    configuration = resolve_experiment("_smoke")
    framework = framework_spec(configuration)
    assert framework.latent_dim == 32
    assert framework.certificate_threshold == pytest.approx(3.0e-12)
    cohort = synthetic_config(configuration)
    assert cohort.shape == (16, 12, 12)
    weights = objective_weights(configuration)
    assert weights.classification == 1.0
    training = training_spec(configuration)
    assert training.epochs == 3
    assert training.world_size == 1
    with pytest.raises(KeyError):
        Configuration({}).require("absent")


def test_build_context_makes_a_usable_experiment() -> None:
    context = build_context("_smoke")
    assert context.grid.shape == (16, 12, 12)
    assert len(context.cohort) > 0
    session = context.session()
    assert session.model.grid.shape == context.grid.shape
    assert repository_root().is_dir()


def test_claim_map_is_internally_consistent() -> None:
    assert len(claim_ids()) == len(CLAIMS)
    assert all(entry["symbols"] for entry in CLAIMS)
    assert all(entry["paper_location"] for entry in CLAIMS)
    mapped = claims_as_mapping(dict.fromkeys(claim_ids(), CheckStatus.PASS.value))
    assert all(entry["status"] == CheckStatus.PASS.value for entry in mapped)
    assert len(mapped) == len(CLAIMS)
    assert code_symbols()
    assert claims_for_module("inverse.certificate")
    statuses = statuses_from_checks({"certificate_thresholding": CheckStatus.PASS.value})
    assert statuses["C9"] == CheckStatus.PASS.value
    assert statuses_from_checks({})["C1"] == CheckStatus.NOT_RUN.value
    assert set(CLAIM_CHECKS) == set(claim_ids())


def test_artefact_writers_are_directed_by_the_root(tmp_path: Path) -> None:
    write_json(tmp_path / "sample.json", {"b": 1})
    write_claim_to_code(tmp_path, {"claims": []})
    write_verification_report(tmp_path, {"overall": "UNVERIFIED", "checks": []})
    write_summary(tmp_path, "summary line")
    write_dataset_urls(tmp_path, ["https://example.invalid/a"])
    assert read_json(tmp_path / CLAIM_NAME)["claims"] == []
    assert read_json(tmp_path / REPORT_NAME)["overall"] == "UNVERIFIED"
    assert (tmp_path / SUMMARY_NAME).read_text().strip() == "summary line"
    assert (tmp_path / DATASET_NAME).read_text().strip() == "https://example.invalid/a"
    payload = write_integrity_manifest(tmp_path)
    assert payload["file_count"] >= 4
    assert MANIFEST_NAME not in payload["files"]
    report = verify_manifest(tmp_path)
    assert report["consistent"]
    (tmp_path / "sample.json").write_text("{}")
    assert not verify_manifest(tmp_path)["consistent"]


def test_status_helpers_summarise_checks() -> None:
    checks = [
        {"name": "a", "status": "PASS"},
        {"name": "b", "status": "NOT_RUN"},
        {"name": "c", "status": "BLOCKED"},
    ]
    counts = status_counts(checks)
    assert counts["PASS"] == 1
    assert overall_verdict(checks) == "PARTIALLY_VERIFIED"
    assert overall_verdict([{"name": "a", "status": "FAIL"}]) == "UNVERIFIED"
    assert overall_verdict([{"name": "a", "status": "PASS"}]) == "VERIFIED"
    table = render_check_table(checks)
    assert "PASS" in table and "BLOCKED" in table
    mapping = render_mapping_table(checks, ("name", "status"), (12, 10))
    assert "name" in mapping


def test_audit_driver_writes_and_reverifies(tmp_path: Path) -> None:
    from mechphase.harness import audit

    repository = Path(__file__).resolve().parents[1]
    writes = audit.write_artefacts(
        tmp_path,
        [
            audit.CheckResult(
                name="driver_probe", status="PASS", evidence="written by the harness test", detail={}
            )
        ],
        {
            "overall": "VERIFIED",
            "claims": [],
            "claim_count": 0,
            "deviations": [],
        },
        elapsed=0.5,
    )
    assert writes["manifest"]["file_count"] >= 4
    report = read_json(tmp_path / REPORT_NAME)
    assert report["overall"] == "VERIFIED"
    assert report["checks"][0]["name"] == "driver_probe"
    assert (tmp_path / SUMMARY_NAME).exists()
    assert verify_manifest(tmp_path)["consistent"]
    assert repository.is_dir()


def test_shipped_artefacts_are_complete() -> None:
    repository = Path(__file__).resolve().parents[1]
    report_path = repository / REPORT_NAME
    claim_path = repository / CLAIM_NAME
    if not report_path.exists():
        pytest.skip("the verification artefacts have not been generated yet")
    report = json.loads(report_path.read_text())
    assert len(report["checks"]) >= 30
    assert report["overall"] in ("VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED")
    assert report["checks"]
    assert all("evidence" in check for check in report["checks"])
    statuses = {check["status"] for check in report["checks"]}
    assert statuses <= {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}
    claims = json.loads(claim_path.read_text())
    assert claims["claim_count"] == len(CLAIMS)
    assert len(claims["claims"]) == len(CLAIMS)
    assert claims["deviations"]
    manifest = json.loads((repository / MANIFEST_NAME).read_text())
    assert manifest["files"]
    assert REPORT_NAME in manifest["files"]
    assert SUMMARY_NAME in manifest["files"]
    assert MANIFEST_NAME not in manifest["files"]
