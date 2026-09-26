"""Paper-claim to code mapping.

Ref: the manuscript's own section, equation, figure and table numbering.

Every entry names a claim, the location in the manuscript it is derived from and the code
symbols that carry it. The verification pass fills in the status.
"""

from __future__ import annotations

from typing import Any

from mechphase.support.types import CheckStatus

CLAIMS: tuple[dict[str, Any], ...] = (
    {
        "id": "C1",
        "statement": "Stage IA adenocarcinoma carries a distinct mechanical phenotype: a higher "
        "lesion-relative modulus ratio, a wider within-lesion ratio spread and a lower loss tangent.",
        "paper_location": "Sec. 2.1, Table 1, Fig. 1",
        "module": "analysis.phenotype, statistics.effectsize",
        "symbols": [
            "src/mechphase/analysis/phenotype.py::phenotype_table",
            "src/mechphase/analysis/phenotype.py::noise_floor_from_repeats",
            "src/mechphase/statistics/effectsize.py::compare_groups",
            "src/mechphase/statistics/effectsize.py::cohens_d",
        ],
    },
    {
        "id": "C2",
        "statement": "The absolute shear modulus is not identifiable from paired phases because "
        "the boundary traction is unmeasured; the admissible estimand is the scale-free pair "
        "(rho, eta).",
        "paper_location": "Sec. 2.2, Proposition 1, Assumptions (A1)-(A3), Sec. 3",
        "module": "constitutive.homogeneity, inverse.identifiability, inverse.boundary",
        "symbols": [
            "src/mechphase/constitutive/homogeneity.py::homogeneity_defect",
            "src/mechphase/constitutive/homogeneity.py::scale_free_estimand",
            "src/mechphase/constitutive/homogeneity.py::gauge_family",
            "src/mechphase/inverse/identifiability.py::scale_invariance_report",
            "src/mechphase/inverse/boundary.py::gauge_from_reference_traction",
        ],
    },
    {
        "id": "C3",
        "statement": "On the identical analysis set the framework reaches a higher area under the "
        "curve than every pre-specified comparator, and the contrast survives the "
        "Holm-Bonferroni adjustment.",
        "paper_location": "Sec. 2.2, Table 2, Fig. 2",
        "module": "estimators.framework, statistics.delong, statistics.multiplicity",
        "symbols": [
            "src/mechphase/estimators/framework.py::MechPhaseModel.forward",
            "src/mechphase/statistics/delong.py::paired_contrast",
            "src/mechphase/statistics/multiplicity.py::holm_bonferroni",
        ],
    },
    {
        "id": "C4",
        "statement": "The increment over the radiomic comparison is largest for the smallest "
        "lesions and for pure ground-glass lesions.",
        "paper_location": "Sec. 2.3, Fig. 3",
        "module": "statistics.strata",
        "symbols": [
            "src/mechphase/statistics/strata.py::assign_size_strata",
            "src/mechphase/statistics/strata.py::assign_morphology_strata",
            "src/mechphase/statistics/strata.py::stratum_increments",
            "src/mechphase/statistics/strata.py::monotone_decreasing",
        ],
    },
    {
        "id": "C5",
        "statement": "Each framework component contributes to the reported area under the curve; "
        "the continuous implicit field and the certificate are within the pre-specified "
        "negligible-difference threshold.",
        "paper_location": "Sec. 2.3, Table 3, Table S2",
        "module": "analysis.ablation, estimators.framework",
        "symbols": [
            "src/mechphase/analysis/ablation.py::ablation_specs",
            "src/mechphase/analysis/ablation.py::evaluate_ablations",
            "src/mechphase/analysis/ablation.py::synergy",
            "src/mechphase/analysis/ablation.py::shuffled_map",
        ],
    },
    {
        "id": "C6",
        "statement": "Assisted readers gain sensitivity on the same lesions and read faster than "
        "unaided readers.",
        "paper_location": "Sec. 2.5, Table 4",
        "module": "analysis.readers",
        "symbols": [
            "src/mechphase/analysis/readers.py::ReaderStudy.paired_sensitivity_gain",
            "src/mechphase/analysis/readers.py::sensitivity_interval",
            "src/mechphase/analysis/readers.py::panel_partition",
        ],
    },
    {
        "id": "C7",
        "statement": "Performance is stable under the pre-specified acquisition and preprocessing "
        "perturbations, with 2.0 mm registration noise the only perturbation reaching the "
        "stability tolerance.",
        "paper_location": "Sec. 2.4, Fig. 4",
        "module": "analysis.perturbation, imaging.registration, imaging.kernel",
        "symbols": [
            "src/mechphase/analysis/perturbation.py::apply_perturbation",
            "src/mechphase/analysis/perturbation.py::evaluate_perturbations",
            "src/mechphase/analysis/perturbation.py::exceeding_tolerance",
        ],
    },
    {
        "id": "C8",
        "statement": "Transfer to unseen sites and regions holds without re-estimation of any "
        "variable or cut-off.",
        "paper_location": "Sec. 2.4, Sec. 4.1",
        "module": "analysis.transportability, cohort.partition",
        "symbols": [
            "src/mechphase/analysis/transportability.py::transfer_report",
            "src/mechphase/analysis/transportability.py::leave_one_site_out",
            "src/mechphase/cohort/partition.py::build_index",
        ],
    },
    {
        "id": "C9",
        "statement": "The identifiability certificate abstains on genuinely harder cases rather "
        "than decoratively.",
        "paper_location": "Sec. 2.2, Sec. 2.7, Fig. 5, Table S5",
        "module": "inverse.certificate, analysis.enumeration",
        "symbols": [
            "src/mechphase/inverse/certificate.py::residual_sensitivity",
            "src/mechphase/inverse/certificate.py::certificate_field",
            "src/mechphase/inverse/certificate.py::certificate_decision",
            "src/mechphase/analysis/enumeration.py::certificate_behaviour",
        ],
    },
    {
        "id": "C10",
        "statement": "Decision-curve analysis gives a net benefit above the radiomic comparator "
        "and above both default strategies across the clinically relevant threshold range.",
        "paper_location": "Sec. 2.7, Fig. 6",
        "module": "statistics.decision",
        "symbols": [
            "src/mechphase/statistics/decision.py::net_benefit",
            "src/mechphase/statistics/decision.py::decision_curve",
            "src/mechphase/statistics/decision.py::exceeds_both_strategies",
        ],
    },
    {
        "id": "C11",
        "statement": "The recovered field is consistent with two references outside the imaging "
        "pipeline: the regional compliance of the respiration-resolved calibration stratum and "
        "the published ex vivo range.",
        "paper_location": "Sec. 4.5, Sec. 2.7",
        "module": "cohort.reference, statistics.agreement",
        "symbols": [
            "src/mechphase/cohort/reference.py::reference_report",
            "src/mechphase/cohort/reference.py::spearman_association",
            "src/mechphase/statistics/agreement.py::icc_two_one",
        ],
    },
    {
        "id": "C12",
        "statement": "In vitro measurements on malignant and non-malignant lung epithelial cell "
        "lines show the same direction of the loss-tangent difference as the imaging phenotype.",
        "paper_location": "Sec. 2.6, Sec. 4.6",
        "module": "analysis.invitro",
        "symbols": [
            "src/mechphase/analysis/invitro.py::in_vitro_report",
            "src/mechphase/analysis/invitro.py::direction_consistency",
            "src/mechphase/analysis/invitro.py::one_way_anova",
        ],
    },
    {
        "id": "C13",
        "statement": "Diagnostic failures fall into four audited classes, each with a named audit.",
        "paper_location": "Sec. 2.2, Table S4",
        "module": "analysis.enumeration, operators.jacobian",
        "symbols": [
            "src/mechphase/analysis/enumeration.py::failure_mode_table",
            "src/mechphase/operators/jacobian.py::violation_count",
            "src/mechphase/inverse/massbalance.py::subsolid_class_b",
        ],
    },
    {
        "id": "C14",
        "statement": "The repeat subset supports the reproducibility floor for both recovered "
        "quantities.",
        "paper_location": "Sec. 2.7, Table S7",
        "module": "statistics.agreement",
        "symbols": [
            "src/mechphase/statistics/agreement.py::agreement_report",
            "src/mechphase/statistics/agreement.py::meets_reproducibility_floor",
            "src/mechphase/analysis/phenotype.py::reproducibility",
        ],
    },
    {
        "id": "C15",
        "statement": "The prospective arm holds 1,247 records partitioned into 382 stage IA, 738 "
        "benign and 127 stage IB-IIIA, and the calibration stratum is excluded from every "
        "diagnostic endpoint.",
        "paper_location": "Sec. 4.1, Table S5",
        "module": "cohort.schema, cohort.partition, cohort.manifest",
        "symbols": [
            "src/mechphase/cohort/schema.py::CohortConstants",
            "src/mechphase/cohort/partition.py::prospective_composition",
            "src/mechphase/cohort/partition.py::assert_reported_totals",
            "src/mechphase/cohort/manifest.py::SyntheticCohort.analysis_indices",
        ],
    },
    {
        "id": "C16",
        "statement": "The statistical test inventory assigns each question its test and, where "
        "pre-specified, its multiplicity correction.",
        "paper_location": "Sec. 4.7, Table S7",
        "module": "analysis.enumeration, statistics.*",
        "symbols": [
            "src/mechphase/analysis/enumeration.py::statistical_test_inventory",
            "src/mechphase/statistics/multiplicity.py::contrast_table",
            "src/mechphase/statistics/roc.py::operating_point",
        ],
    },
    {
        "id": "C17",
        "statement": "Compute, wall-clock and emissions accounting is reported honestly for the "
        "training and inference accelerators.",
        "paper_location": "Table S8",
        "module": "analysis.enumeration, loop.parallel",
        "symbols": [
            "src/mechphase/analysis/enumeration.py::compute_accounting",
            "src/mechphase/loop/parallel.py::device_summary",
        ],
    },
)


def claim_ids() -> tuple[str, ...]:
    return tuple(str(claim["id"]) for claim in CLAIMS)


def claims_as_mapping(statuses: dict[str, str] | None = None) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for claim in CLAIMS:
        entry = dict(claim)
        entry["status"] = (
            statuses.get(str(claim["id"]), CheckStatus.NOT_RUN.value)
            if statuses
            else CheckStatus.NOT_RUN.value
        )
        payload.append(entry)
    return payload


def code_symbols() -> list[str]:
    symbols: list[str] = []
    for claim in CLAIMS:
        symbols.extend(str(item) for item in claim["symbols"])
    return symbols


def claims_for_module(module: str) -> list[str]:
    return [str(claim["id"]) for claim in CLAIMS if module in str(claim["module"])]


CLAIM_CHECKS: dict[str, tuple[str, ...]] = {
    "C1": ("paper_claim_mapping", "size_strata_partition"),
    "C2": (
        "constant_phase_constitutive_relations",
        "stress_scale_homogeneity",
        "manufactured_equilibrium_solution",
        "scale_non_identifiability",
        "scale_free_estimand_invariance",
    ),
    "C3": ("auc_independent_library", "delong_variance_independent", "delong_paired_difference_independent", "holm_bonferroni_independent"),
    "C4": ("size_strata_partition", "operating_point_independent", "analysis_layer_execution"),
    "C5": ("component_switches_execution", "loss_and_backward_execution"),
    "C6": ("analysis_layer_execution", "single_batch_overfit_execution"),
    "C7": ("framework_forward_execution", "checkpoint_round_trip_execution"),
    "C8": ("analysis_layer_execution", "reported_constants_match_manuscript"),
    "C9": ("certificate_thresholding", "certificate_sensitivity_closed_form"),
    "C10": ("net_benefit_independent", "operating_point_independent", "analysis_layer_execution"),
    "C11": ("spearman_independent", "icc_two_one_independent", "analysis_layer_execution"),
    "C12": ("analysis_layer_execution",),
    "C13": ("jacobian_determinant_closed_form", "mass_coupling_closure", "interior_residual_floor"),
    "C14": ("icc_two_one_independent",),
    "C15": ("cohort_schema_compatibility", "reported_constants_match_manuscript"),
    "C16": ("holm_bonferroni_independent", "operating_point_independent", "analysis_layer_execution"),
    "C17": ("reported_constants_match_manuscript", "analysis_layer_execution"),
}


def statuses_from_checks(check_status: dict[str, str]) -> dict[str, str]:
    """Fold the check outcomes into a per-claim status."""

    statuses: dict[str, str] = {}
    for claim_id, names in CLAIM_CHECKS.items():
        observed = [check_status.get(name) for name in names]
        known = [value for value in observed if value is not None]
        if not known:
            statuses[claim_id] = CheckStatus.NOT_RUN.value
        elif any(value == CheckStatus.FAIL.value for value in known):
            statuses[claim_id] = CheckStatus.FAIL.value
        elif any(value == CheckStatus.NOT_RUN.value for value in known):
            statuses[claim_id] = CheckStatus.NOT_RUN.value
        elif any(value == CheckStatus.BLOCKED.value for value in known):
            statuses[claim_id] = CheckStatus.BLOCKED.value
        else:
            statuses[claim_id] = CheckStatus.PASS.value
    return statuses
