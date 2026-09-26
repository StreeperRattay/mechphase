"""Verification driver: runs every check, writes the root artefacts, then re-verifies.

Ref: the manuscript's reported tables and the release's own claim map.

Artefact order matters. The claim map, the report, the plain-text summary and the dataset
links are written first, then the integrity manifest covers the final tree including those
files and excluding only itself. After the manifest is written the live tree is re-hashed
and diffed against it, and an inconsistent manifest is a hard failure of the run.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mechphase.harness import verification as verify
from mechphase.harness.artefacts import (
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
from mechphase.harness.claims import claims_as_mapping, statuses_from_checks
from mechphase.harness.verification import CheckResult
from mechphase.support.logging import configure_logging, get_logger
from mechphase.support.types import CheckStatus, OverallVerdict

LOGGER = get_logger("harness.audit")

SCOPE = "local source, generated stand-in cohort and closed-form references"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def dependency_versions() -> dict[str, str]:
    import numpy
    import scipy
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def run_checks(
    root: Path, scratch: Path, run_tools: bool, probe_links: bool = False
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks.append(verify.check_claim_mapping(root))
    checks.append(verify.check_verified_symbols(root))
    checks.append(verify.check_reported_constants())

    checks.append(verify.check_constant_phase_relations())
    checks.append(verify.check_stress_scale_homogeneity())
    checks.append(verify.check_manufactured_equilibrium())
    checks.append(verify.check_scale_non_identifiability())
    checks.append(verify.check_jacobian_closed_form())
    checks.append(verify.check_mass_coupling_closure())
    checks.append(verify.check_interior_residual_floor())
    checks.append(verify.check_scale_free_estimand())
    checks.append(verify.check_certificate_thresholding())
    checks.append(verify.check_certificate_sensitivity())

    checks.append(verify.check_auc_against_library())
    checks.append(verify.check_delong_variance())
    checks.append(verify.check_delong_paired())
    checks.append(verify.check_holm_bonferroni())
    checks.append(verify.check_icc())
    checks.append(verify.check_operating_point())
    checks.append(verify.check_net_benefit())
    checks.append(verify.check_standardised_difference())
    checks.append(verify.check_spearman())
    checks.append(verify.check_size_strata_partition())
    checks.append(verify.check_radiomic_descriptors())

    checks.append(verify.check_cohort_schema())
    checks.append(verify.check_generator_closure())
    checks.append(verify.check_analysis_layer())
    checks.append(verify.check_experiment_execution())
    checks.append(verify.check_component_switches())
    checks.append(verify.check_gradient_flow())
    checks.append(verify.check_parameter_update())
    checks.append(verify.check_checkpoint_round_trip(scratch))
    checks.append(verify.check_single_batch_overfit())
    checks.append(verify.check_minimal_training_loop(scratch))
    checks.append(verify.check_public_dataset_links(probe=probe_links))

    if run_tools:
        checks.append(
            verify.check_tool("ruff", [sys.executable, "-m", "ruff", "check", "."], root)
        )
        checks.append(
            verify.check_tool(
                "mypy_strict",
                [sys.executable, "-m", "mypy", "--strict", "src/mechphase"],
                root,
            )
        )
        checks.append(
            verify.check_tool("pytest_suite", [sys.executable, "-m", "pytest", "-q"], root)
        )
    else:
        checks.append(verify.not_run("ruff", "tool checks were disabled for this run"))
        checks.append(verify.not_run("mypy_strict", "tool checks were disabled for this run"))
        checks.append(verify.not_run("pytest_suite", "tool checks were disabled for this run"))

    checks.append(verify.check_docker())
    checks.append(verify.check_private_cohorts())
    checks.append(verify.check_pretrained_weights())
    return checks



def build_report(checks: Sequence[CheckResult], manifest_note: str) -> dict[str, Any]:
    payloads = [check.as_mapping() for check in checks]
    counts = status_counts(payloads)
    outstanding = [
        check.evidence
        for check in checks
        if check.status in (CheckStatus.NOT_RUN.value, CheckStatus.BLOCKED.value)
    ]
    failed = [check.as_mapping() for check in checks if check.status == CheckStatus.FAIL.value]
    return {
        "overall": (
            OverallVerdict.VERIFIED.value
            if not failed and not outstanding
            else (
                OverallVerdict.UNVERIFIED.value
                if failed
                else OverallVerdict.PARTIALLY_VERIFIED.value
            )
        ),
        "scope": SCOPE,
        "checks": payloads,
        "status_counts": counts,
        "failed_checks": failed,
        "unverified_reasons": outstanding,
        "dependency_versions": dependency_versions(),
        "integrity_manifest": manifest_note,
    }

def build_claim_map(checks: Sequence[CheckResult]) -> dict[str, Any]:
    from mechphase.harness.claims import CLAIM_CHECKS, CLAIMS

    check_status = {check.name: check.status for check in checks}
    statuses = statuses_from_checks(check_status)
    entries = claims_as_mapping(statuses)
    deviations = [
        {
            "topic": "isotropic part of the stress",
            "paper_location": "Assumption (A1), Proposition 1",
            "departure": "The bulk response is written as a multiple of the shear modulus rather "
            "than as a fixed bulk modulus, so that the whole stress is homogeneous of degree one "
            "in the modulus, which is exactly the condition Proposition 1 relies on.",
            "justification": "A fixed bulk modulus breaks sigma(cG, eta) = c sigma(G, eta).",
        },
        {
            "topic": "residual domain",
            "paper_location": "Assumption (A3)",
            "departure": "The residual energy and the mass-coupling term are evaluated with a "
            "one-voxel boundary shell excluded.",
            "justification": "The boundary traction is unmeasured, and the one-sided stencil at "
            "the shell is not the interior equilibrium equation.",
        },
        {
            "topic": "stand-in cohort",
            "paper_location": "Sec. 4.1",
            "departure": "A schema-compatible generator stands in for the private cohorts. Its "
            "deformation is a uniaxial state whose axial stiffness product is constant along the "
            "deformation axis, the modulus is defined on the same stencil the residual uses, and "
            "the expiratory density follows from mass conservation.",
            "justification": "The clinical cohorts are private; the generator makes every "
            "mechanism executable and keeps the mechanical ground truth in closed form.",
        },
        {
            "topic": "loss tangent of the stand-in cohort",
            "paper_location": "Sec. 4.1",
            "departure": "The generated loss tangent is constant per record by default, with an "
            "optional smooth ripple.",
            "justification": "For the uniaxial ansatz the imaginary part of the equilibrium "
            "condition forces the loss tangent to be constant along the deformation axis, so a "
            "spatially varying tangent would leave a residual floor rather than an exact state.",
        },
        {
            "topic": "objective weights",
            "paper_location": "Sec. 4.5",
            "departure": "The relative weights of the objective terms are engineering defaults "
            "exposed in the training configuration.",
            "justification": "The manuscript states the objective's components but does not print "
            "the weights.",
        },
        {
            "topic": "certificate threshold",
            "paper_location": "Sec. 2.2, Table S5",
            "departure": "The certificate threshold and ridge are engineering defaults exposed in "
            "the model configuration.",
            "justification": "The manuscript reports the abstention rate and the certified "
            "fractions but not the numeric threshold of the Fisher statistic.",
        },
        {
            "topic": "development-arm sizes and voxel spacing",
            "paper_location": "Sec. 4.1, Sec. 4.2",
            "departure": "The per-site counts of the development arm and the voxel spacing are "
            "engineering defaults supplied by configuration.",
            "justification": "The manuscript reports the prospective and external site counts and "
            "the arm composition but not the development counts or the spacing.",
        },
        {
            "topic": "clinical instrument coefficients",
            "paper_location": "Table 2, Sec. 4.4",
            "departure": "The Brock/PanCan risk model and the Lung-RADS lexicon are implemented "
            "with coefficient vectors and a category table that are engineering defaults.",
            "justification": "The manuscript cites the instruments but does not reproduce their "
            "coefficients.",
        },
        {
            "topic": "encoder weights",
            "paper_location": "Sec. 4.4, Code availability statement",
            "departure": "The encoder architecture and its masked-reconstruction objective ship "
            "with the release; no pretrained checkpoint is bundled and pretrained_path is null.",
            "justification": "The thoracic CT checkpoint is not redistributable.",
        },
        {
            "topic": "comparator training",
            "paper_location": "Sec. 4.4, Table 2",
            "departure": "The comparator implementations share the schedule, augmentations and "
            "data split of the framework but are exercised on the stand-in cohort only.",
            "justification": "The reported comparator areas were measured on the private cohorts, "
            "which are not available here; the comparator rows are therefore reported as not run.",
        },
    ]
    return {
        "overall": "VERIFIED"
        if all(entry["status"] == CheckStatus.PASS.value for entry in entries)
        else "PARTIALLY_VERIFIED",
        "claims": entries,
        "claim_checks": {key: list(value) for key, value in CLAIM_CHECKS.items()},
        "deviations": deviations,
        "claim_count": len(CLAIMS),
        "code_symbol_count": sum(len(claim["symbols"]) for claim in CLAIMS),
    }



def render_summary(
    checks: Sequence[CheckResult], claims: dict[str, Any], verdict: str
) -> str:
    counts = status_counts([check.as_mapping() for check in checks])
    lines = [
        "mechphase verification summary",
        "==============================",
        "",
        f"overall                     {verdict}",
        f"scope                       {SCOPE}",
        f"checks                      {len(checks)}",
        "",
        "Checks",
        "------",
        render_check_table([check.as_mapping() for check in checks]),
        "",
        "Status counts",
        "-------------",
    ]
    for key in ("PASS", "FAIL", "NOT_RUN", "BLOCKED"):
        lines.append(f"{key.ljust(28)} {counts.get(key, 0)}")
    lines.extend(["", "Outstanding items", "-----------------"])
    outstanding = [check for check in checks if check.status in ("NOT_RUN", "BLOCKED")]
    if outstanding:
        for check in outstanding:
            lines.append(f"- {check.name}: {check.evidence}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Claim-to-code map",
            "=================",
            "",
            f"overall                     {claims['overall']}",
            f"claims                      {claims['claim_count']}",
            "",
            render_mapping_table(
                [
                    {
                        "claim": entry["id"],
                        "status": entry["status"],
                        "paper": entry["paper_location"],
                    }
                    for entry in claims["claims"]
                ],
                ("claim", "status", "paper"),
                (16, 30, 40),
            ),
            "",
            "Deviations",
            "----------",
        ]
    )
    for deviation in claims["deviations"]:
        lines.append(f"- {deviation['topic']}: {deviation['departure']}")
    return "\n".join(lines) + "\n"


def write_artefacts(
    root: Path, checks: Sequence[CheckResult], claims: dict[str, Any], elapsed: float
) -> dict[str, Any]:
    manifest_note = (
        "integrity_manifest.json is written after this report and after the plain-text summary, "
        "covers every tracked file including this report, and excludes only itself"
    )
    report = build_report(checks, manifest_note)
    write_dataset_urls(root, verify.recorded_dataset_urls())
    write_claim_to_code(root, claims)
    write_verification_report(root, report)
    write_summary(root, render_summary(checks, claims, str(report["overall"])))
    manifest = write_integrity_manifest(root)
    # The duration is returned, never written: the report and the summary are digested by the
    # manifest, so a wall-clock value inside them would make the release fail to reproduce itself.
    return {"report": report, "manifest": manifest, "elapsed_seconds": round(elapsed, 3)}

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the release verification pass")
    parser.add_argument("--root", type=Path, default=None, help="repository root to audit")
    parser.add_argument("--skip-tools", action="store_true", help="skip ruff, mypy and pytest")
    parser.add_argument(
        "--probe-links",
        action="store_true",
        help="probe the recorded collection links from this host instead of leaving it NOT_RUN",
    )
    parser.add_argument("--log-level", default="INFO")
    arguments = parser.parse_args(argv)
    configure_logging(arguments.log_level)
    root = repository_root() if arguments.root is None else arguments.root.resolve()
    scratch = Path(tempfile.mkdtemp(prefix="mechphase-audit-"))
    start = time.perf_counter()
    checks = run_checks(
        root, scratch, run_tools=not arguments.skip_tools, probe_links=arguments.probe_links
    )
    claims = build_claim_map(checks)
    elapsed = time.perf_counter() - start
    payload = write_artefacts(root, checks, claims, elapsed)
    live = verify_manifest(root)
    report = payload["report"]
    LOGGER.info("overall %s", report["overall"])
    LOGGER.info("checks %s", status_counts([check.as_mapping() for check in checks]))
    LOGGER.info("elapsed %.3f s", payload["elapsed_seconds"])
    if not live["consistent"]:
        LOGGER.error("integrity manifest is inconsistent with the live tree: %s", live)
        return 2
    LOGGER.info("integrity manifest consistent over %d files", live["recorded_files"])
    print(json.dumps({"overall": report["overall"], "checks": len(checks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
