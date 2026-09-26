"""Verification-driver tests.

The writers take an explicit root, so these tests exercise the driver against a temporary
directory. One test additionally asserts that the shipped artefacts are the complete ones,
because a driver that writes to a temporary root still has to leave the shipped report
untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mechphase.harness import verification as verify

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_closed_form_checks_pass() -> None:
    results = [
        verify.check_constant_phase_relations(),
        verify.check_stress_scale_homogeneity(),
        verify.check_manufactured_equilibrium(),
        verify.check_scale_non_identifiability(),
        verify.check_jacobian_closed_form(),
        verify.check_interior_residual_floor(),
        verify.check_scale_free_estimand(),
        verify.check_certificate_sensitivity(),
    ]
    for result in results:
        assert result.status == "PASS", (result.name, result.evidence)


def test_static_checks_pass() -> None:
    results = [
        verify.check_claim_mapping(REPOSITORY_ROOT),
        verify.check_verified_symbols(REPOSITORY_ROOT),
        verify.check_auc_against_library(),
        verify.check_delong_variance(),
        verify.check_delong_paired(),
        verify.check_holm_bonferroni(),
        verify.check_icc(),
        verify.check_operating_point(),
        verify.check_net_benefit(),
        verify.check_standardised_difference(),
        verify.check_spearman(),
        verify.check_size_strata_partition(),
        verify.check_radiomic_descriptors(),
        verify.check_certificate_thresholding(),
        verify.check_cohort_schema(),
        verify.check_generator_closure(),
        verify.check_analysis_layer(),
        verify.check_reported_constants(),
        verify.check_mass_coupling_closure(),
    ]
    for result in results:
        assert result.status == "PASS", (result.name, result.evidence)


def test_execution_checks_pass(tmp_path: Path) -> None:
    results = [
        verify.check_experiment_execution(),
        verify.check_component_switches(),
        verify.check_gradient_flow(),
        verify.check_parameter_update(),
        verify.check_checkpoint_round_trip(tmp_path),
        verify.check_single_batch_overfit(),
        verify.check_minimal_training_loop(tmp_path),
    ]
    for result in results:
        assert result.status == "PASS", (result.name, result.evidence)


def test_unavailable_checks_are_reported_honestly() -> None:
    assert verify.check_docker().status == "BLOCKED"
    assert verify.check_private_cohorts().status == "NOT_RUN"
    assert verify.check_pretrained_weights().status == "BLOCKED"


def test_driver_writes_to_the_requested_root(tmp_path: Path) -> None:
    from mechphase.harness import audit

    checks = [
        verify.CheckResult(
            name="static", status="PASS", evidence="checked in the driver test", detail={}
        ),
        verify.not_run("unavailable", "not exercised by this test"),
    ]
    claims = audit.build_claim_map(checks)
    payload = audit.write_artefacts(tmp_path, checks, claims, elapsed=0.25)
    assert payload["report"]["overall"] == "PARTIALLY_VERIFIED"
    assert (tmp_path / "verification_report.json").exists()
    assert (tmp_path / "claim_to_code.json").exists()
    assert (tmp_path / "verification_summary.txt").exists()
    assert (tmp_path / "dataset_urls.txt").exists()
    assert verify.check_manifest_consistency(tmp_path).status == "PASS"
    manifest = json.loads((tmp_path / "integrity_manifest.json").read_text())
    assert "verification_report.json" in manifest["files"]
    assert "integrity_manifest.json" not in manifest["files"]


def test_shipped_report_is_the_complete_one() -> None:
    report_path = REPOSITORY_ROOT / "verification_report.json"
    if not report_path.exists():
        pytest.skip("the shipped artefacts have not been generated yet")
    report = json.loads(report_path.read_text())
    names = {check["name"] for check in report["checks"]}
    assert len(names) >= 30
    for required in (
        "manufactured_equilibrium_solution",
        "scale_non_identifiability",
        "certificate_sensitivity_closed_form",
        "minimal_training_loop_execution",
        "single_batch_overfit_execution",
        "checkpoint_round_trip_execution",
        "ruff",
        "mypy_strict",
        "pytest_suite",
    ):
        assert required in names, required
    assert report["overall"] in ("VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED")
