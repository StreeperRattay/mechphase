"""Statistics tests: discrimination, DeLong, multiplicity, resampling and decision analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mechphase.statistics.agreement import (
    AgreementReport,
    agreement_report,
    between_within_ratio,
    icc_two_one,
    mean_squares,
    meets_reproducibility_floor,
    repeatability_coefficient,
    spearman,
)
from mechphase.statistics.decision import (
    DecisionCurve,
    covers_radiomics,
    decision_curve,
    exceeds_both_strategies,
    net_benefit,
    reported_point,
    score_at_threshold,
    treat_all_net_benefit,
    treat_none_net_benefit,
)
from mechphase.statistics.delong import (
    PairedContrast,
    delong,
    interval_for,
    paired_contrast,
    placement_values,
    reproduce_area,
)
from mechphase.statistics.effectsize import (
    GroupSummary,
    cohens_d,
    compare_groups,
    mannwhitney,
    median_quartile_range,
    standardised_mean_difference,
    stratified_comparisons,
    summarise,
)
from mechphase.statistics.multiplicity import (
    adjustment_summary,
    benjamini_hochberg,
    bonferroni,
    contrast_table,
    family_size_from_components,
    holm_bonferroni,
    significant,
    smallest_adjusted,
)
from mechphase.statistics.resampling import (
    auc_bootstrap_interval,
    bootstrap_indices,
    bootstrap_interval,
    bootstrap_statistic,
    paired_bootstrap_difference,
    stratified_bootstrap_indices,
)
from mechphase.statistics.roc import (
    RocCurve,
    binary_decisions,
    binormal_curve,
    confusion_counts,
    increment,
    interval_from_bootstrap,
    material_change,
    midranks,
    operating_point,
    pointwise_intervals,
    rates_from_counts,
    roc_auc,
    roc_curve,
    sensitivity_at_specificity,
)
from mechphase.statistics.strata import (
    MORPHOLOGY_ORDER,
    SIZE_ORDER,
    assign_morphology_strata,
    assign_size_strata,
    counts_by_group,
    groups_from_labels,
    monotone_decreasing,
    partition_or_raise,
    pooled_increment,
    rare_strata,
    sample_distribution,
    stratum_increments,
)
from mechphase.support.types import Interval, Morphology


def _binary_problem(seed: int = 7, size: int = 120, effect: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    labels = np.concatenate([np.ones(size // 2), np.zeros(size - size // 2)])
    scores = generator.normal(size=size) + effect * labels
    return scores, labels


def test_roc_auc_matches_the_rank_definition() -> None:
    scores, labels = _binary_problem()
    positive = scores[labels == 1.0]
    negative = scores[labels == 0.0]
    manual = np.mean(
        [(1.0 if p > n else 0.5 if p == n else 0.0) for p in positive for n in negative]
    )
    assert roc_auc(scores, labels) == pytest.approx(float(manual), abs=1e-12)
    with pytest.raises(ValueError):
        roc_auc(scores, labels + 1.0)
    with pytest.raises(ValueError):
        roc_auc(scores, np.ones_like(labels))


def test_midranks_average_over_ties() -> None:
    values = np.asarray([1.0, 2.0, 2.0, 4.0])
    ranks = midranks(values)
    assert ranks[1] == pytest.approx(2.5)
    assert ranks[2] == pytest.approx(2.5)
    assert ranks[3] == pytest.approx(4.0)


def test_roc_curve_and_operating_point() -> None:
    scores, labels = _binary_problem()
    curve = roc_curve(scores, labels)
    assert isinstance(curve, RocCurve)
    assert curve.thresholds.size == curve.sensitivity.size == curve.specificity.size
    assert float(curve.false_positive_rate.min()) >= 0.0
    point = operating_point(scores, labels)
    assert 0.0 <= point.sensitivity <= 1.0
    assert 0.0 <= point.specificity <= 1.0
    closest = operating_point(scores, labels, method="closest")
    assert 0.0 <= closest.sensitivity <= 1.0
    with pytest.raises(ValueError):
        operating_point(scores, labels, method="unsupported")
    lower, upper = pointwise_intervals(
        curve.sensitivity[:5], curve.specificity[:5], 60, 60
    )
    assert bool((lower <= 1.0).all())
    assert bool((upper <= 1.0).all())
    target = sensitivity_at_specificity(scores, labels, 0.8)
    assert 0.0 <= target <= 1.0
    counts = confusion_counts(scores, labels, point.threshold)
    sensitivity, specificity = rates_from_counts(counts)
    assert sensitivity == pytest.approx(point.sensitivity, abs=1e-12)
    assert specificity == pytest.approx(point.specificity, abs=1e-12)
    assert binary_decisions(scores, 0.0).shape == scores.shape


def test_binormal_curve_and_increment_helpers() -> None:
    false_positive, sensitivity = binormal_curve(0.9)
    assert false_positive.shape == sensitivity.shape
    assert float(sensitivity[-1]) > float(sensitivity[0])
    with pytest.raises(ValueError):
        binormal_curve(1.0)
    assert increment(0.94, 0.871) == pytest.approx(0.069)
    assert material_change(0.04, 0.03)
    assert not material_change(0.02, 0.03)
    interval = interval_from_bootstrap(np.asarray([0.9, 0.92, 0.94]))
    assert interval.low <= interval.high


def test_delong_matches_a_double_loop() -> None:
    scores, labels = _binary_problem(seed=11)
    result = delong(scores, labels)
    positive = scores[labels == 1.0]
    negative = scores[labels == 0.0]
    matrix = np.zeros((positive.size, negative.size))
    for row, value in enumerate(positive):
        for column, other in enumerate(negative):
            matrix[row, column] = 1.0 if value > other else (0.5 if value == other else 0.0)
    v10 = matrix.mean(axis=1)
    v01 = matrix.mean(axis=0)
    expected = np.var(v10, ddof=1) / positive.size + np.var(v01, ddof=1) / negative.size
    assert result.variance == pytest.approx(float(expected), rel=1e-12)
    assert result.area == pytest.approx(float(v10.mean()), rel=1e-12)
    assert isinstance(result.interval(), Interval)
    assert reproduce_area(scores, labels) == pytest.approx(result.area)
    assert interval_for(scores, labels).low <= result.area


def test_paired_contrast_is_finite() -> None:
    scores, labels = _binary_problem(seed=13)
    other = scores + 0.2 * labels
    contrast = paired_contrast(scores, other, labels)
    assert isinstance(contrast, PairedContrast)
    assert contrast.standard_error > 0.0
    assert 0.0 <= contrast.p_value <= 1.0
    first, second = placement_values(scores, labels)
    assert first.size + second.size == labels.size


def test_multiplicity_adjustments_are_monotone() -> None:
    raw = np.asarray([0.001, 0.02, 0.3, 0.6])
    holm = holm_bonferroni(raw)
    assert bool((holm >= raw - 1e-15).all())
    assert bool((holm <= 1.0).all())
    bh = benjamini_hochberg(raw)
    assert bool((bh >= raw - 1e-15).all())
    assert bool((bonferroni(raw) >= raw - 1e-15).all())
    contrasts = contrast_table(["a", "b", "c", "d"], [1.0, 2.0, 3.0, 4.0], raw.tolist())
    assert len(contrasts) == 4
    assert significant(contrasts, alpha=0.05)
    assert smallest_adjusted(contrasts) <= holm.max()
    assert adjustment_summary(contrasts)["count"] == 4.0
    assert family_size_from_components(comparator=8, phenotype=3, subgroup=2) == 13
    assert adjustment_summary([]) == {"count": 0.0}
    assert holm_bonferroni(np.asarray([])).size == 0


def test_resampling_helpers_are_deterministic() -> None:
    values = np.linspace(0.0, 1.0, 50)
    generator = np.random.default_rng(3)
    indices = bootstrap_indices(values.size, 10, generator)
    assert indices.shape == (10, 50)
    interval = bootstrap_interval(values, np.mean)
    assert interval.point == pytest.approx(0.5)
    assert interval.high >= interval.low
    draws = bootstrap_statistic(values, np.mean, resamples=50, seed=1)
    assert draws.shape == (50,)
    paired = paired_bootstrap_difference(values, values - 0.2, resamples=50, seed=1)
    assert paired.shape == (50,)
    assert float(np.mean(paired)) == pytest.approx(0.2, abs=1e-9)
    stratified = stratified_bootstrap_indices(
        np.asarray([0, 0, 1, 1, 1]), 5, np.random.default_rng(2)
    )
    assert stratified.shape == (5, 5)
    scores, labels = _binary_problem(size=60)
    auc_interval = auc_bootstrap_interval(scores, labels, resamples=40, seed=1)
    assert auc_interval.low <= auc_interval.high
    with pytest.raises(ValueError):
        bootstrap_indices(0, 4, np.random.default_rng(1))


def test_effect_size_helpers() -> None:
    generator = np.random.default_rng(5)
    first = generator.normal(0.4, 1.0, size=40)
    second = generator.normal(0.0, 1.0, size=60)
    assert cohens_d(first, second) > 0.0
    assert standardised_mean_difference(first, second) > 0.0
    assert 0.0 <= mannwhitney(first, second) <= 1.0
    summary = summarise(first)
    assert isinstance(summary, GroupSummary)
    assert summary.count == 40
    assert summary.interquartile >= 0.0
    assert median_quartile_range(first) >= 0.0
    comparison = compare_groups(first, second, resamples=100, seed=1)
    assert comparison.difference.low <= comparison.difference.high
    assert "effect_size" in comparison.as_mapping()
    from mechphase.statistics.effectsize import orientation

    assert orientation(comparison, comparison)
    payload = stratified_comparisons({"first": (first, second)}, resamples=50, seed=1)
    assert "first" in payload
    with pytest.raises(ValueError):
        cohens_d(np.asarray([1.0]), second)
    with pytest.raises(ValueError):
        summarise(np.asarray([]))


def test_agreement_helpers_recover_a_known_icc() -> None:
    generator = np.random.default_rng(9)
    subjects = 30
    latent = generator.normal(size=subjects)
    matrix = np.stack([latent + generator.normal(scale=0.5, size=subjects) for _ in range(3)], axis=1)
    report = agreement_report(matrix)
    assert isinstance(report, AgreementReport)
    assert 0.0 <= report.icc <= 1.0
    assert report.within_subject_cv >= 0.0
    assert meets_reproducibility_floor(report, floor=0.0)
    assert repeatability_coefficient(matrix) >= 0.0
    assert between_within_ratio(matrix) > 0.0
    msr, msc, mse = mean_squares(matrix)
    assert msr > 0.0 and msc >= 0.0 and mse > 0.0
    with pytest.raises(ValueError):
        icc_two_one(np.zeros((1, 3)))
    assert spearman(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
    assert math.isnan(spearman(np.arange(2.0), np.arange(2.0)))
    with pytest.raises(ValueError):
        spearman(np.arange(3.0), np.arange(4.0))


def test_decision_curve_analysis() -> None:
    assert net_benefit(0.9, 0.9, 0.5, 0.25) == pytest.approx(
        0.9 * 0.5 - 0.1 * 0.5 * (0.25 / 0.75)
    )
    assert treat_all_net_benefit(0.5, 0.25) == pytest.approx(0.5 - 0.5 * (0.25 / 0.75))
    assert treat_none_net_benefit() == 0.0
    with pytest.raises(ValueError):
        net_benefit(0.9, 0.9, 0.5, 1.0)
    with pytest.raises(ValueError):
        treat_all_net_benefit(0.5, -0.1)
    scores, labels = _binary_problem(size=200, effect=1.2)
    curve = decision_curve(scores, labels)
    assert isinstance(curve, DecisionCurve)
    assert curve.thresholds.size == curve.net_benefit.size
    point = reported_point(curve, 0.2)
    assert set(point) == {"threshold", "net_benefit", "treat_all", "treat_none", "advantage"}
    assert isinstance(exceeds_both_strategies(curve), bool)
    reference = decision_curve(scores - 0.5, labels)
    assert isinstance(covers_radiomics(curve, reference), bool)
    with pytest.raises(ValueError):
        covers_radiomics(curve, decision_curve(scores, labels, thresholds=np.asarray([0.1])))
    detail = score_at_threshold(scores, labels, 0.2)
    assert set(detail) == {"sensitivity", "specificity", "prevalence", "net_benefit", "auc"}


def test_stratification_helpers() -> None:
    diameters = [3.0, 8.0, 15.0, 25.0, 9.0]
    labels = assign_size_strata(diameters)
    groups = groups_from_labels(labels)
    assert sum(int(np.count_nonzero(mask)) for mask in groups.values()) == len(diameters)
    partition_or_raise(labels, "size")
    morphologies = assign_morphology_strata(
        [Morphology.SOLID.value, Morphology.GROUND_GLASS.value]
    )
    assert set(morphologies.tolist()) == {Morphology.SOLID.value, Morphology.GROUND_GLASS.value}
    with pytest.raises(ValueError):
        assign_morphology_strata(["unknown"])
    targets = np.asarray([1, 0, 1, 0, 1])
    counts = counts_by_group(labels, targets)
    assert sum(item.total for item in counts.values()) == len(diameters)
    scores, outcome = _binary_problem(size=60)
    reference = scores - 0.3
    widths = np.linspace(4.5, 24.0, 60)
    increments = stratum_increments(scores, reference, outcome, assign_size_strata(widths.tolist()))
    assert set(increments) <= set(SIZE_ORDER)
    assert isinstance(monotone_decreasing(increments, SIZE_ORDER), bool)
    assert set(MORPHOLOGY_ORDER) == {
        Morphology.GROUND_GLASS.value,
        Morphology.PART_SOLID.value,
        Morphology.SOLID.value,
    }
    assert rare_strata(np.asarray(["a", "a", "b"]), minimum=2) == ["b"]
    distribution = sample_distribution(np.asarray(["a", "a", "b", "b"]))
    assert distribution["a"] == pytest.approx(0.5)
    weighted = pooled_increment({"a": 0.2, "b": 0.4}, {"a": 1.0, "b": 3.0})
    assert weighted == pytest.approx(0.35)
    with pytest.raises(ValueError):
        pooled_increment({}, {})
