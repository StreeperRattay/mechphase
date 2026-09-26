"""Analysis-layer tests: phenotype, ablations, perturbations, transfer, readers and in vitro."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mechphase.analysis.ablation import (
    ABLATION_SPECS,
    REPORTED_ABLATIONS,
    AblationOutcome,
    ablation_specs,
    certificate_ablation_effect,
    evaluate_ablations,
    regression_against_reported,
    reported_synergy,
    shuffled_map,
    synergy,
    unreported_configurations,
)
from mechphase.analysis.enumeration import (
    CertificateBehaviour,
    FailureModeTable,
    certificate_behaviour,
    certificate_by_outcome,
    compute_accounting,
    failure_mode_table,
    inventory_by_question,
    jacobian_failures,
    requires_correction,
    residual_failures,
    statistical_test_inventory,
)
from mechphase.analysis.invitro import (
    CELL_LINES,
    CellLine,
    cell_line,
    direction_consistency,
    in_vitro_report,
    malignant_lines,
    non_malignant_lines,
    one_way_anova,
    pooled_malignant,
    pooled_non_malignant,
    protocol,
    stiffness_consistency,
    synthesise_replicates,
    tukey_between,
)
from mechphase.analysis.perturbation import (
    REPORTED_PERTURBATIONS,
    Perturbation,
    apply_perturbation,
    degraded_segmentation_mask,
    evaluate_perturbations,
    exceeding_tolerance,
    kernel_sensitivity,
    pre_specified,
    registration_curve,
    smoothing_only,
    worst_case,
)
from mechphase.analysis.phenotype import (
    PhenotypeTable,
    heterogeneity,
    noise_floor_from_repeats,
    phenotype_table,
    recovery_error,
    reproducibility,
    split_by_outcome,
    subtract_noise_floor,
)
from mechphase.analysis.readers import (
    PANEL_LESIONS,
    READER_COUNT,
    ReaderArm,
    ReaderStudy,
    arm_from_calls,
    panel_partition,
    ranked_arms,
    sensitivity_interval,
    summarise_study,
)
from mechphase.analysis.transportability import (
    TransferReport,
    drift_within_tolerance,
    leave_one_site_out,
    region_shift,
    split_scores,
    transfer_report,
    vendor_offsets_applied,
)
from mechphase.cohort.partition import build_index
from mechphase.support.types import Outcome


def test_phenotype_table_matches_the_reported_direction() -> None:
    generator = np.random.default_rng(3)
    outcomes = [Outcome.STAGE_IA] * 40 + [Outcome.BENIGN] * 60
    ratio = np.concatenate([generator.normal(1.42, 0.2, 40), generator.normal(1.08, 0.2, 60)])
    tangent = np.concatenate([generator.normal(0.118, 0.02, 40), generator.normal(0.171, 0.02, 60)])
    spread = np.concatenate([generator.normal(0.83, 0.1, 40), generator.normal(0.46, 0.1, 60)])
    table = phenotype_table(ratio, tangent, spread, outcomes, noise_floor=0.05, resamples=200)
    assert isinstance(table, PhenotypeTable)
    payload = table.as_mapping()
    assert payload["modulus_ratio"]["difference"] > 0.0
    assert payload["loss_tangent"]["difference"] < 0.0
    assert table.strongest_effect() in payload
    assert table.all_p_below(alpha=0.01)
    assert set(table.directions()) == set(payload)


def test_phenotype_helpers() -> None:
    malignant, benign = split_by_outcome(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        [Outcome.STAGE_IA, Outcome.BENIGN, Outcome.STAGE_IA, Outcome.BENIGN],
    )
    assert malignant.tolist() == [1.0, 3.0]
    assert benign.tolist() == [2.0, 4.0]
    adjusted = subtract_noise_floor(np.asarray([0.5, 0.1]), 0.3)
    assert adjusted.tolist() == [pytest.approx(0.2), pytest.approx(0.0)]
    assert noise_floor_from_repeats(np.asarray([1.0, 2.0]), np.asarray([1.1, 1.8])) == pytest.approx(0.15)
    assert heterogeneity(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx(2.0)
    error = recovery_error(np.asarray([1.1, 1.2]), np.asarray([1.0, 1.0]))
    assert error["bias"] > 0.0
    with pytest.raises(ValueError):
        recovery_error(np.asarray([1.0]), np.asarray([1.0, 2.0]))
    matrix = np.stack(
        [np.linspace(0.0, 1.0, 20), np.linspace(0.0, 1.0, 20) + 0.05, np.linspace(0.0, 1.0, 20) - 0.05],
        axis=1,
    )
    report = reproducibility(matrix[:, 0], matrix[:, 1])
    assert 0.0 <= report["icc_2_1"] <= 1.0


def test_ablation_helpers() -> None:
    from mechphase.estimators.framework import FrameworkSpec

    specs = ablation_specs(FrameworkSpec())
    assert set(specs) == {"full"} | {name for name, _ in ABLATION_SPECS}
    assert len(unreported_configurations()) == len(REPORTED_ABLATIONS) * (len(REPORTED_ABLATIONS) - 1) // 2
    outcomes = evaluate_ablations(0.940, {"without_cpr": 0.874, "without_fic": 0.944})
    assert all(isinstance(item, AblationOutcome) for item in outcomes)
    material = next(item for item in outcomes if item.name == "without_cpr")
    assert material.material
    assert material.components_removed == ("use_constant_phase_residual",)
    assert not next(item for item in outcomes if item.name == "without_fic").material
    assert synergy(0.940, 0.891, 0.874, 0.861) == pytest.approx(0.036)
    assert reported_synergy() == pytest.approx(0.036, abs=1e-9)
    assert set(REPORTED_ABLATIONS) >= {"morphology_only", "shuffled_map"}
    payload = regression_against_reported({"without_cpr": 0.874}, tolerance=0.01)
    assert payload["without_cpr"]["within_tolerance"] == 1.0
    effect = certificate_ablation_effect(np.asarray([0.71] * 5), np.asarray([0.951] * 5))
    assert effect["gap"] > 0.0


def test_shuffled_map_preserves_the_values(record) -> None:
    lesion = record.fields["lesion"]
    shuffled = shuffled_map(record.modulus, lesion, torch.Generator().manual_seed(1))
    assert torch.allclose(torch.sort(shuffled[lesion].flatten()).values, torch.sort(record.modulus[lesion].flatten()).values)
    assert torch.allclose(shuffled[~lesion], record.modulus[~lesion])
    empty = torch.zeros_like(lesion)
    assert torch.equal(shuffled_map(record.modulus, empty), record.modulus)


def test_perturbation_pipeline_runs(records) -> None:
    assert len(pre_specified()) == len(REPORTED_PERTURBATIONS)
    baseline = 0.94
    payload = evaluate_perturbations(
        records,
        evaluate=lambda _items: baseline,
        baseline=baseline,
        seed=1,
    )
    assert set(payload) == {item.name for item in REPORTED_PERTURBATIONS}
    name, delta = worst_case(payload)
    assert name in payload
    assert delta == pytest.approx(0.0)
    assert exceeding_tolerance(payload) == []
    mutated = {key: {"auc": value["auc"], "delta": value["delta"]} for key, value in payload.items()}
    mutated["registration_2.0mm"]["auc"] = baseline - 0.05
    mutated["registration_2.0mm"]["delta"] = -0.05
    assert exceeding_tolerance(mutated) == ["registration_2.0mm"]
    assert worst_case(mutated)[0] == "registration_2.0mm"
    record = records[0]
    for perturbation in REPORTED_PERTURBATIONS:
        perturbed = apply_perturbation(record, perturbation, seed=3)
        assert set(perturbed.fields) == set(record.fields)
    with pytest.raises(ValueError):
        apply_perturbation(record, Perturbation("unknown", "unknown", 0.0), seed=1)
    _mask, dice = degraded_segmentation_mask(record.fields["lesion"], 0.85)
    assert 0.0 < dice <= 1.0
    assert smoothing_only(record.fields["lesion"], 1.0).shape == record.fields["lesion"].shape
    assert kernel_sensitivity(record)
    assert registration_curve(record, [0.0, 1.0], record.grid)


def test_transportability_reports(records) -> None:
    entries = build_index(development_per_site=1)
    subset = entries[: len(records)]
    scores = np.linspace(0.0, 1.0, len(records))
    labels = np.asarray([record.label if record.label is not None else 0 for record in records], dtype=float)
    labels[:2] = 0.0
    labels[2:] = 1.0
    splits = split_scores(scores, labels, subset)
    assert isinstance(splits, dict)
    report = transfer_report(scores, labels, subset)
    assert isinstance(report, TransferReport)
    assert set(report.as_mapping()) >= {"development", "external_gap"}
    assert isinstance(drift_within_tolerance(report, tolerance=1.0), bool)
    assert isinstance(leave_one_site_out(scores, labels, subset), dict)
    assert isinstance(region_shift(scores, labels, subset), dict)
    adjusted = vendor_offsets_applied(scores, [0.1, 0.2])
    assert adjusted.shape == scores.shape
    assert np.allclose(vendor_offsets_applied(scores, []), scores)
    with pytest.raises(ValueError):
        split_scores(scores[:1], labels, subset)


def test_failure_mode_and_certificate_tables(records) -> None:
    table = failure_mode_table(records)
    assert isinstance(table, FailureModeTable)
    payload = table.as_mapping()
    assert payload["mass_conservation_violation"]["count"] >= 0.0
    assert table.total() >= 0
    assert jacobian_failures(records) >= 0
    assert residual_failures(records) >= 0
    scores = np.linspace(0.0, 1.0, len(records))
    labels = np.asarray([record.label for record in records], dtype=float)
    fractions = np.asarray([record.certified_fraction for record in records])
    certified = (fractions > 0.5).astype(float)
    behaviour = certificate_behaviour(scores, labels, certified, fractions)
    assert isinstance(behaviour, CertificateBehaviour)
    assert behaviour.all_cases == len(records)
    assert 0.0 <= behaviour.as_mapping()["abstention_rate"] <= 1.0
    assert isinstance(behaviour.certificate_is_informative(), bool)
    assert set(certificate_by_outcome(fractions, labels)) == {
        "median_fraction_benign",
        "median_fraction_stage_ia",
    }
    with_outcome = failure_mode_table(records, predicted=scores, labels=labels)
    assert with_outcome.partition_of(int(np.count_nonzero(certified < 0.5)))


def test_enumeration_tables() -> None:
    inventory = statistical_test_inventory()
    assert len(inventory) == 6
    assert inventory[0]["test"] == "DeLong"
    assert inventory_by_question()["reproducibility"] == "ICC(2,1)"
    assert requires_correction("comparator AUC differences")
    assert not requires_correction("independent-reference association")
    accounting = compute_accounting()
    assert accounting["training_accelerators"] == 4.0
    assert accounting["inference_seconds_per_examination"] == 6.2
    assert accounting["estimated_emissions_kg_co2e"] == 41.0


def test_in_vitro_report_is_consistent() -> None:
    report = in_vitro_report()
    assert report["anova_p_value"] < 0.01
    assert report["modulus_direction_consistent"] == 1.0
    assert report["tangent_direction_consistent"] == 1.0
    assert stiffness_consistency()
    assert direction_consistency(0.118, 0.171)
    assert not direction_consistency(0.2, 0.1)
    lines = synthesise_replicates(cell_line("A549"), seed=1)
    assert lines.size == 3 * 30
    statistic, p_value = one_way_anova([lines, synthesise_replicates(cell_line("BEAS-2B"), seed=2)])
    assert statistic > 0.0
    assert 0.0 <= p_value <= 1.0
    tukey_statistic, tukey_probability = tukey_between(lines, lines)
    assert 0.0 <= tukey_probability <= 1.0
    assert tukey_statistic == pytest.approx(0.0, abs=1e-9)
    assert len(malignant_lines()) == 2
    assert len(non_malignant_lines()) == 2
    assert pooled_malignant({line.name: lines for line in CELL_LINES}).size == 2 * lines.size
    assert pooled_non_malignant({line.name: lines for line in CELL_LINES}).size == 2 * lines.size
    assert protocol()["biological_replicates"] == 3.0
    with pytest.raises(KeyError):
        cell_line("unknown")
    assert isinstance(CELL_LINES[0], CellLine)


def test_reader_study_helpers() -> None:
    unaided = ReaderArm("unaided", 0.862, 0.748, 0.862, 118.0, 157, 181)
    assisted = ReaderArm("assisted", 0.913, 0.833, 0.895, 96.0, 175, 188)
    framework = ReaderArm("framework", 0.934, 0.886, 0.933, 6.2, 186, 196)
    study = ReaderStudy()
    for arm in (unaided, assisted, framework):
        study.add(arm)
    assert isinstance(study.as_mapping(), dict)
    assert study.paired_sensitivity_gain() == pytest.approx(8.5, abs=1e-9)
    assert study.exceeds_clinical_difference()
    assert study.reading_time_change() < 0.0
    assert study.by_name("framework").seconds_per_case == pytest.approx(6.2)
    assert ranked_arms(study)[0] == "framework"
    assert summarise_study(study)["readers"] == float(READER_COUNT)
    assert summarise_study(ReaderStudy()) == {}
    partition = panel_partition()
    assert sum(partition.values()) == PANEL_LESIONS
    with pytest.raises(ValueError):
        panel_partition(1, 1)
    with pytest.raises(KeyError):
        study.by_name("absent")
    scores = np.concatenate([np.linspace(0.8, 1.0, 20), np.linspace(0.0, 0.4, 20)])
    labels = np.concatenate([np.ones(20), np.zeros(20)])
    arm = arm_from_calls("derived", scores, labels, 0.5, 10.0)
    assert arm.sensitivity == pytest.approx(1.0)
    assert arm.specificity == pytest.approx(1.0)
    interval = sensitivity_interval(16, 20, 14, 20, resamples=50, seed=1)
    assert interval.low <= interval.high


def test_split_scores_requires_alignment() -> None:
    entries = build_index(development_per_site=1)[:3]
    with pytest.raises(ValueError):
        split_scores(np.asarray([1.0]), np.asarray([1.0]), entries)
