"""Scoring entry point.

Ref: Table 2 (discrimination on the analysis set), Fig. 3 (the stratified increment),
Fig. 5 (the certificate subgroups), Fig. 6 (the decision curve).

The reported cohort-level endpoints cannot be recomputed because the clinical cohorts are
private. This entry point therefore scores whatever cohort is configured, reports the
certificate composition and the decision curve, and records the cohort the numbers came
from so that a stand-in cohort is never mistaken for the analysis set.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mechphase.analysis.enumeration import certificate_behaviour
from mechphase.cohort.batching import batches_from
from mechphase.estimators.framework import MechPhaseModel
from mechphase.harness.context import ExperimentContext, build_context, split_positions
from mechphase.loop.checkpointing import load_checkpoint
from mechphase.statistics.decision import decision_curve, reported_point
from mechphase.statistics.strata import (
    MORPHOLOGY_ORDER,
    SIZE_ORDER,
    assign_morphology_strata,
    assign_size_strata,
    stratum_increments,
)
from mechphase.support.config import repository_root
from mechphase.support.io import write_text
from mechphase.support.logging import configure_logging, get_logger
from mechphase.support.types import Split

LOGGER = get_logger("harness.score")


@dataclass(slots=True)
class ScoreBundle:
    score: np.ndarray
    label: np.ndarray
    fraction: np.ndarray
    certified: np.ndarray
    ratio: np.ndarray
    tangent: np.ndarray
    diameter: np.ndarray
    morphology: np.ndarray
    model: torch.nn.Module


def score_positions(context: ExperimentContext, positions: Sequence[int]) -> ScoreBundle:
    records = [context.cohort[position] for position in positions]
    model = MechPhaseModel(context.grid, context.framework)
    model.eval()
    scores: list[float] = []
    labels: list[float] = []
    fractions: list[float] = []
    certified: list[float] = []
    ratio: list[float] = []
    tangent: list[float] = []
    diameters: list[float] = []
    morphologies: list[str] = []
    with torch.no_grad():
        for batch in batches_from(records, context.training.batch_size):
            outputs = model(dict(batch))
            scores.extend(outputs["logits"].reshape(-1).tolist())
            labels.extend(batch["label"].reshape(-1).tolist())
            fractions.extend(outputs["certified_fraction"].reshape(-1).tolist())
            certified.extend((outputs["certified_fraction"] > 0.0).reshape(-1).tolist())
    for record in records:
        ratio.append(record.true_pair.rho)
        tangent.append(record.true_pair.eta)
        diameters.append(record.diameter_mm)
        morphologies.append(record.morphology.value)
    return ScoreBundle(
        score=np.asarray(scores, dtype=np.float64),
        label=np.asarray(labels, dtype=np.float64),
        fraction=np.asarray(fractions, dtype=np.float64),
        certified=np.asarray(certified, dtype=np.float64),
        ratio=np.asarray(ratio, dtype=np.float64),
        tangent=np.asarray(tangent, dtype=np.float64),
        diameter=np.asarray(diameters, dtype=np.float64),
        morphology=np.asarray(morphologies),
        model=model,
    )


def render_report(name: str, cohort: str, payload: ScoreBundle) -> str:
    labels = payload.label
    keep = labels >= 0
    scores = payload.score[keep]
    targets = payload.label[keep]
    fractions = payload.fraction[keep]
    certified = payload.certified[keep]
    lines = [
        f"experiment                  {name}",
        f"cohort                      {cohort}",
        f"cases                       {int(targets.size)}",
        "",
    ]
    if targets.size and np.unique(targets).size == 2:
        composition = certificate_behaviour(scores, targets, certified, fractions)
        curve = decision_curve(scores, targets)
        lines.extend(["discrimination", "--------------"])
        for key, value in sorted(composition.as_mapping().items()):
            lines.append(f"{key.ljust(28)} {value:.6f}")
        lines.extend(["", "decision curve at the reported threshold", "---------------------------------------"])
        for key, value in sorted(reported_point(curve).items()):
            lines.append(f"{key.ljust(28)} {value:.6f}")
        by_size = assign_size_strata(payload.diameter.tolist())
        by_morphology = assign_morphology_strata(payload.morphology.tolist())
        ratios = np.abs(payload.ratio - 1.0)
        lines.extend(["", "stratified increment against the ratio reference", "-----------------------------------------------"])
        for label, strata in (("size", by_size), ("morphology", by_morphology)):
            increments = stratum_increments(scores, ratios, targets, strata)
            for key in sorted(increments):
                lines.append(f"{label}.{key.ljust(24)} {increments[key]:.6f}")
        lines.extend(
            [
                "",
                "stratum orders used by the monitoring of the increment",
                f"size        {' > '.join(SIZE_ORDER)}",
                f"morphology  {' > '.join(MORPHOLOGY_ORDER)}",
            ]
        )
    else:
        lines.append("discrimination               not computable: the cohort is single-class")
    lines.extend(
        [
            "",
            "note",
            "----",
            "the clinical cohorts are private, so this report describes the configured stand-in "
            "cohort and must not be read as the reported analysis-set values",
        ]
    )
    return "\n".join(lines) + "\n"



def run(
    experiment: str,
    config_root: Path | None,
    overrides: Sequence[str],
    limit: int,
    output: Path | None,
) -> Path:
    context = build_context(experiment, config_root, tuple(overrides))
    positions = split_positions(context, Split.PROSPECTIVE, limit)
    if not positions:
        positions = split_positions(context, Split.DEVELOPMENT, limit)
    checkpoint = (output or repository_root() / "runs") / f"{experiment}.pt"
    payload = score_positions(context, positions)
    if checkpoint.exists():
        load_checkpoint(checkpoint, payload.model, restore_rng=False)
    destination = (output or repository_root() / "runs") / f"{experiment}_score.txt"
    write_text(destination, render_report(experiment, "stand-in cohort", payload))
    LOGGER.info("wrote %s", destination.name)
    return destination

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="score the framework on the configured cohort")
    parser.add_argument("--experiment", default="main")
    parser.add_argument("--config-root", type=Path, default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    arguments = parser.parse_args(argv)
    configure_logging(arguments.log_level)
    run(
        arguments.experiment,
        arguments.config_root,
        tuple(arguments.override),
        arguments.limit,
        arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
