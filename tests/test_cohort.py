"""Cohort tests: schema, partitioning, the stand-in generator, indices and reference."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tests.conftest import DEVELOPMENT_SITE, SMALL_SHAPE, make_record

from mechphase.cohort.batching import (
    batches_from,
    class_balance,
    collate,
    labelled_subset,
    spec_of,
    to_device,
    volume_shapes,
)
from mechphase.cohort.manifest import (
    CohortManifest,
    SyntheticCohort,
    expected_reported_counts,
    sample_positions,
)
from mechphase.cohort.partition import (
    CLASS_ORDER,
    analysis_indices,
    arm_totals,
    assert_reported_totals,
    assign_folds,
    build_index,
    external_composition,
    index_counts,
    largest_remainder,
    partition_summary,
    prospective_analysis_indices,
    prospective_composition,
    repeat_subset,
    size_stratum_of,
    stratify_by_morphology,
    stratify_by_route,
    stratify_by_size,
    training_seeds,
    within_split_indices,
)
from mechphase.cohort.reference import (
    ReferenceReport,
    calibration_stratum_summary,
    coverage_fraction,
    ex_vivo_range,
    published_ex_vivo_interval,
    reference_report,
    reference_verdict,
    regional_compliance_values,
    spearman_association,
)
from mechphase.cohort.schema import (
    ANALYSIS_SET_COUNT,
    BENIGN_COUNT,
    CALIBRATION_TOTAL,
    PROSPECTIVE_TOTAL,
    REPEAT_SUBSET_COUNT,
    STAGE_IA_COUNT,
    STAGE_IB_IIIA_COUNT,
    CohortConstants,
    SiteSpec,
    all_sites,
    required_field_names,
    site_layout,
    validate_analysis_counts,
    validate_prospective_total,
)
from mechphase.cohort.synthetic import (
    DEFAULT_SPACING_MM,
    MORPHOLOGY_DENSITY,
    PHENOTYPE_PRIORS,
    SyntheticConfig,
    axial_coefficient,
    build_record,
    calibration_reference,
    closure_report,
    ellipsoid_indicator,
    modulus_from_strain,
    realised_pair,
    rescale_modulus,
    sample_morphology,
    sample_phenotype,
    smooth_texture,
)
from mechphase.support.types import (
    Certification,
    LabelRoute,
    Morphology,
    Outcome,
    SizeStratum,
    Split,
)


def test_site_layout_matches_the_reported_counts() -> None:
    layout = site_layout()
    assert layout["Site D"] == 486
    assert layout["Site E"] == 421
    assert layout["Site F"] == 340
    assert layout["Site G"] == 448
    assert layout["Site H"] == 332
    assert layout["Calibration"] == CALIBRATION_TOTAL
    assert sum(layout[name] for name in ("Site D", "Site E", "Site F")) == PROSPECTIVE_TOTAL
    assert len(all_sites()) == 9
    assert required_field_names()[0] == "inspiration"


def test_cohort_constants_and_validators() -> None:
    constants = CohortConstants()
    assert constants.as_mapping()["analysis_set"] == float(ANALYSIS_SET_COUNT)
    validate_analysis_counts(STAGE_IA_COUNT, BENIGN_COUNT)
    validate_prospective_total(STAGE_IA_COUNT, BENIGN_COUNT, STAGE_IB_IIIA_COUNT)
    with pytest.raises(ValueError):
        validate_analysis_counts(1, 1)
    with pytest.raises(ValueError):
        validate_prospective_total(1, 1, 1)


def test_largest_remainder_preserves_the_total() -> None:
    shares = largest_remainder([3.0, 1.0, 1.0], 100)
    assert sum(shares) == 100
    assert shares[0] > shares[1]
    assert largest_remainder([], 5) == []
    assert sum(largest_remainder([0.0, 0.0], 7)) == 7
    with pytest.raises(ValueError):
        largest_remainder([1.0], -1)


def test_arm_compositions_reproduce_the_reported_classes() -> None:
    prospective = prospective_composition()
    totals = dict.fromkeys(CLASS_ORDER, 0)
    for site, shares in prospective.items():
        assert sum(shares.values()) == site.count
        for outcome, value in shares.items():
            totals[outcome] += value
    assert totals[Outcome.STAGE_IA] == STAGE_IA_COUNT
    assert totals[Outcome.BENIGN] == BENIGN_COUNT
    assert totals[Outcome.STAGE_IB_IIIA] == STAGE_IB_IIIA_COUNT
    external = external_composition()
    assert sum(shares[Outcome.STAGE_IA] for shares in external.values()) == round(
        780 * STAGE_IA_COUNT / ANALYSIS_SET_COUNT
    )


def test_build_index_matches_the_reported_arms() -> None:
    entries = build_index(development_per_site=6)
    assert_reported_totals(entries)
    totals = arm_totals(entries)
    assert totals[Split.PROSPECTIVE.value] == PROSPECTIVE_TOTAL
    assert totals[Split.EXTERNAL_VALIDATION.value] == 780
    assert totals[Split.CALIBRATION.value] == CALIBRATION_TOTAL
    analysis = analysis_indices(entries)
    assert len(analysis) == ANALYSIS_SET_COUNT + 780 + 3 * 6
    assert all(entries[position].site.split is not Split.CALIBRATION for position in analysis)
    prospective = prospective_analysis_indices(entries)
    assert len(prospective) == ANALYSIS_SET_COUNT
    assert within_split_indices(entries, Split.CALIBRATION)
    counts = index_counts(entries)
    assert counts["outcome::stage_IA"] >= STAGE_IA_COUNT
    assert counts["outcome::benign"] >= BENIGN_COUNT
    assert counts["split::external_validation"] == 780


def test_stratification_and_subsampling_helpers() -> None:
    diameters = [3.0, 8.0, 15.0, 25.0]
    groups = stratify_by_size(diameters)
    assert sum(len(members) for members in groups.values()) == 4
    assert groups[SizeStratum.LE_6MM] == [0]
    summary = partition_summary(groups, 4)
    assert summary["unassigned"] == 0.0
    morphologies = [Morphology.SOLID, Morphology.GROUND_GLASS]
    morphology_groups = stratify_by_morphology(morphologies)
    assert len(morphology_groups[Morphology.SOLID]) == 1
    routes = [LabelRoute.HISTOPATHOLOGY, LabelRoute.FOLLOW_UP]
    route_groups = stratify_by_route(routes)
    assert len(route_groups[LabelRoute.FOLLOW_UP]) == 1
    folds = assign_folds(10, 5, seed=1)
    assert sorted(folds) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert len(repeat_subset(400, seed=1, size=REPEAT_SUBSET_COUNT)) == REPEAT_SUBSET_COUNT
    with pytest.raises(ValueError):
        repeat_subset(10, seed=1)
    assert training_seeds(5)[0] == 23
    assert size_stratum_of(4.0) is SizeStratum.LE_6MM
    with pytest.raises(ValueError):
        assign_folds(4, 1, seed=0)
    with pytest.raises(ValueError):
        training_seeds(0)


def test_generator_produces_a_closed_form_state() -> None:
    config = SyntheticConfig(shape=SMALL_SHAPE)
    record = make_record(3, Outcome.STAGE_IA, config)
    assert record.true_pair.rho > 0.0
    assert record.true_pair.eta > 0.0
    assert record.heterogeneity >= 0.0
    assert record.certificate in Certification
    assert record.morphology in Morphology
    assert record.diameter_mm > 0.0
    assert record.label in (0, 1)
    assert record.stratum() in {stratum.value for stratum in SizeStratum}
    residual_max = record.residual(shell=0).magnitude_max()
    stress_scale = float(torch.max(record.stress().magnitude()).item())
    assert residual_max < 0.06 * stress_scale
    closure = closure_report(record)
    assert closure["truth_energy"] < closure["spoiled_energy"]
    assert closure["ratio"] > 2.0
    assert record.phenotype_mapping()["rho"] == pytest.approx(record.true_pair.rho)


def test_generator_is_deterministic_and_reproducible() -> None:
    config = SyntheticConfig(shape=SMALL_SHAPE)
    first = make_record(5, Outcome.BENIGN, config)
    second = make_record(5, Outcome.BENIGN, config)
    assert first.true_pair.rho == pytest.approx(second.true_pair.rho)
    assert torch.allclose(first.fields["inspiration"], second.fields["inspiration"])
    other = make_record(6, Outcome.BENIGN, config)
    assert not torch.equal(first.fields["lesion"], other.fields["lesion"])


def test_phenotype_sampling_respects_the_reported_ordering() -> None:
    generator = torch.Generator().manual_seed(11)
    benign = [sample_phenotype(Outcome.BENIGN, generator)[0] for _ in range(200)]
    malignant = [sample_phenotype(Outcome.STAGE_IA, generator)[0] for _ in range(200)]
    benign_rho = np.mean([pair.rho for pair in benign])
    malignant_rho = np.mean([pair.rho for pair in malignant])
    benign_eta = np.mean([pair.eta for pair in benign])
    malignant_eta = np.mean([pair.eta for pair in malignant])
    assert malignant_rho > benign_rho
    assert malignant_eta < benign_eta
    assert set(PHENOTYPE_PRIORS) == {Outcome.BENIGN, Outcome.STAGE_IA, Outcome.STAGE_IB_IIIA}
    drawn = {sample_morphology(generator) for _ in range(200)}
    assert drawn == set(Morphology)


def test_generator_geometry_helpers() -> None:
    config = SyntheticConfig(shape=SMALL_SHAPE)
    spec = config.grid()
    indicator = ellipsoid_indicator(spec, (8.0, 6.0, 6.0), (3.0, 3.0, 3.0))
    assert float(indicator.max().item()) == pytest.approx(1.0, abs=1e-6)
    assert float(indicator.min().item()) >= 0.0
    texture = smooth_texture(spec, torch.Generator().manual_seed(1), 0.1)
    assert float(texture.abs().max().item()) <= 0.1 + 1e-9
    assert float(smooth_texture(spec, torch.Generator().manual_seed(1), 0.0).abs().max().item()) == 0.0
    coefficient = axial_coefficient(0.2, 10.0)
    assert coefficient > 10.0
    modulus = modulus_from_strain(torch.full((1, *SMALL_SHAPE), 0.01, dtype=torch.float64), 1.0, coefficient)
    assert float(modulus.min().item()) > 0.0
    assert MORPHOLOGY_DENSITY[Morphology.SOLID] > MORPHOLOGY_DENSITY[Morphology.GROUND_GLASS]
    assert DEFAULT_SPACING_MM[0] == 1.0


def test_rescale_modulus_only_touches_the_region(record) -> None:
    lesion = record.fields["lesion"]
    scaled = rescale_modulus(record.modulus, 2.0, lesion)
    assert torch.allclose(scaled[~lesion], record.modulus[~lesion])
    assert torch.allclose(scaled[lesion], 2.0 * record.modulus[lesion])


def test_realised_pair_matches_the_closed_form(record) -> None:
    pair = realised_pair(
        record.modulus, record.loss_tangent, record.fields["lesion"], record.fields["parenchyma"]
    )
    assert pair.rho == pytest.approx(record.true_pair.rho)
    assert pair.eta == pytest.approx(record.true_pair.eta)


def test_collation_and_shapes(records) -> None:
    spec = spec_of(records)
    assert spec.shape == SMALL_SHAPE
    batch = collate(records)
    assert batch["inspiration"].shape == (4, *SMALL_SHAPE)
    assert batch["displacement"].shape == (4, 3, *SMALL_SHAPE)
    assert batch["label"].tolist() == [1, 0, 1, 0]
    assert to_device(batch, torch.device("cpu"))["label"].tolist() == [1, 0, 1, 0]
    assert len(labelled_subset(records)) == 4
    balance = class_balance(records)
    assert balance["positives"] == 2
    assert volume_shapes(records)["lesion"] == (1, *SMALL_SHAPE)
    assert len(batches_from(records, 3)) == 2
    with pytest.raises(ValueError):
        collate([])
    with pytest.raises(ValueError):
        batches_from(records, 0)
    with pytest.raises(ValueError):
        spec_of([make_record(0), build_record("other", DEVELOPMENT_SITE, Outcome.BENIGN, SyntheticConfig(shape=(8, 8, 8)), 1)])


def test_cohort_sequence_and_manifest() -> None:
    config = SyntheticConfig(shape=SMALL_SHAPE, development_per_site=2)
    cohort = SyntheticCohort(config, entries=build_index(2))
    assert len(cohort) == len(cohort.entries)
    first = cohort[0]
    assert first.record_id == cohort.entries[0].record_id
    assert cohort[-1].record_id == cohort.entries[-1].record_id
    assert [record.record_id for record in cohort[0:2]] == [
        cohort.entries[0].record_id,
        cohort.entries[1].record_id,
    ]
    assert cohort.indices_for_split(Split.PROSPECTIVE)
    manifest = cohort.manifest()
    assert isinstance(manifest, CohortManifest)
    assert manifest.as_mapping()["development_per_site"] == 2.0
    assert expected_reported_counts()["analysis_set"] == ANALYSIS_SET_COUNT
    with pytest.raises(IndexError):
        cohort[len(cohort) + 1]
    positions = sample_positions(cohort, 3)
    assert len(positions) <= 3


def test_reference_analysis_is_well_formed() -> None:
    config = SyntheticConfig(shape=SMALL_SHAPE)
    site = SiteSpec(name="Calibration", region="Region I", count=8, split=Split.CALIBRATION)
    records = [
        build_record(f"cal-{index}", site, Outcome.BENIGN, config, index) for index in range(6)
    ]
    report = reference_report(records)
    assert isinstance(report, ReferenceReport)
    assert report.samples == 6
    compliance = regional_compliance_values(records)
    assert compliance.shape == (6,)
    moduli = np.asarray([float(torch.median(record.modulus).item()) for record in records])
    assert np.isfinite(spearman_association(compliance, moduli))
    low, high = ex_vivo_range(records)
    assert high >= low
    assert coverage_fraction(moduli, low, high) > 0.0
    assert published_ex_vivo_interval() == (1.0, 6.0)
    assert isinstance(reference_verdict(report), bool)
    assert calibration_stratum_summary(records)["records"] == 6.0
    reference = calibration_reference(records[0], torch.Generator().manual_seed(1))
    assert set(reference) == {
        "regional_compliance",
        "reference_eta",
        "ex_vivo_modulus_low",
        "ex_vivo_modulus_high",
    }
    with pytest.raises(ValueError):
        reference_report([])
