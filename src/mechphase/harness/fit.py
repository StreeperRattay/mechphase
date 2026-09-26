"""Training entry point.

Ref: Sec. 4.5 (training details), Sec. 4.1 (the internal split), Sec. 4.7 (the seed sweep).

The run writes a plain-text training report under the run directory. Nothing in the report
is invented: every number comes from the session that just executed.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mechphase.cohort.batching import batches_from
from mechphase.harness.context import build_context, split_positions
from mechphase.loop.session import TrainingHistory, TrainingSession, evaluate
from mechphase.support.config import repository_root
from mechphase.support.io import write_text
from mechphase.support.logging import configure_logging, get_logger
from mechphase.support.types import Split

LOGGER = get_logger("harness.fit")



def render_report(
    name: str, session: TrainingSession, history: TrainingHistory, validation: dict[str, float]
) -> str:
    lines = [
        f"experiment                  {name}",
        "objective and schedule",
        "----------------------",
    ]
    for key in sorted(session.describe()):
        lines.append(f"{key.ljust(28)} {session.describe()[key]}")
    lines.extend(["", "objective weights", "-----------------"])
    for key, weight in sorted(session.weights.as_mapping().items()):
        lines.append(f"{key.ljust(28)} {weight}")
    lines.extend(["", "run", "---"])
    lines.append(f"epochs_completed            {history.epochs}")
    lines.append(f"optimiser_steps             {history.steps}")
    if history.loss:
        lines.append(f"initial_loss                {history.loss[0]:.6f}")
        lines.append(f"final_loss                  {history.loss[-1]:.6f}")
    else:
        lines.append("initial_loss                n/a")
        lines.append("final_loss                  n/a")
    lines.append(f"largest_parameter_change    {history.parameter_change:.6e}")
    if history.checkpoint is not None:
        for key, entry in sorted(history.checkpoint.as_mapping().items()):
            lines.append(f"checkpoint_{key.ljust(18)} {entry}")
    lines.extend(["", "final validation", "----------------"])
    for key in sorted(validation):
        lines.append(f"{key.ljust(28)} {validation[key]}")
    lines.extend(["", "loss trajectory (first and last five)", "------------------------------------"])
    lines.append(f"head {[round(value, 6) for value in history.loss[:5]]}")
    lines.append(f"tail {[round(value, 6) for value in history.loss[-5:]]}")
    return "\n".join(lines) + "\n"


def run(
    experiment: str,
    config_root: Path | None,
    overrides: Sequence[str],
    train_limit: int,
    valid_limit: int,
    output: Path | None,
) -> dict[str, float]:
    context = build_context(experiment, config_root, tuple(overrides))
    train_positions = split_positions(context, Split.DEVELOPMENT, train_limit)
    valid_positions = split_positions(context, Split.INTERNAL_VALIDATION, valid_limit)
    if not valid_positions:
        valid_positions = split_positions(context, Split.PROSPECTIVE, valid_limit)
    records = [context.cohort[position] for position in train_positions]
    validation_records = [context.cohort[position] for position in valid_positions]
    checkpoint = (output or repository_root() / "runs") / f"{experiment}.pt"
    session = context.session(checkpoint)
    history = session.fit(
        batches_from(records, context.training.batch_size),
        batches_from(validation_records, context.training.batch_size),
    )
    validation = evaluate(
        session.model, batches_from(validation_records, context.training.batch_size)
    )
    destination = (output or repository_root() / "runs") / f"{experiment}_report.txt"
    write_text(destination, render_report(experiment, session, history, validation))
    LOGGER.info("wrote %s", destination.name)
    return history.as_mapping()

def parse_overrides(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="train the framework on a configured split")
    parser.add_argument("--experiment", default="main")
    parser.add_argument("--config-root", type=Path, default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--validation-limit", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    arguments = parser.parse_args(argv)
    configure_logging(arguments.log_level)
    summary = run(
        arguments.experiment,
        arguments.config_root,
        parse_overrides(arguments.override),
        arguments.train_limit,
        arguments.validation_limit,
        arguments.output,
    )
    for key in sorted(summary):
        LOGGER.info("%-28s %s", key, summary[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
