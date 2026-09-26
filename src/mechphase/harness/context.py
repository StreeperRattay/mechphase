"""Configuration-to-object resolution shared by the command-line entry points.

Ref: Sec. 4.5 (training details), Sec. 4.1 (cohort composition).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mechphase.cohort.batching import spec_of
from mechphase.cohort.manifest import SyntheticCohort
from mechphase.cohort.synthetic import SyntheticConfig, build_record
from mechphase.estimators.framework import FrameworkSpec, MechPhaseModel
from mechphase.loop.averaging import AveragingSpec
from mechphase.loop.optimise import OptimiserSpec
from mechphase.loop.precision import PrecisionSpec
from mechphase.loop.schedule import ScheduleSpec, steps_per_epoch, total_steps
from mechphase.loop.session import TrainingSession, TrainingSpec
from mechphase.objectives.weights import ObjectiveWeights
from mechphase.support.config import Configuration, resolve_experiment
from mechphase.support.types import GridSpec, Split


@dataclass(slots=True)
class ExperimentContext:
    name: str
    configuration: Configuration
    grid: GridSpec
    framework: FrameworkSpec
    weights: ObjectiveWeights
    training: TrainingSpec
    cohort: SyntheticCohort

    def session(self, checkpoint: Path | None = None) -> TrainingSession:
        model = MechPhaseModel(self.grid, self.framework)
        return TrainingSession(model, self.training, self.weights, self.framework, checkpoint)


def framework_spec(configuration: Configuration) -> FrameworkSpec:
    section = configuration.section("framework")
    return FrameworkSpec(
        bulk_ratio=section.as_float("bulk_ratio", 10.0),
        boundary_shell=section.as_int("boundary_shell", 1),
        certificate_threshold=section.as_float("certificate_threshold", 3.0e-12),
        certificate_ridge=section.as_float("certificate_ridge", 1.0e-8),
        certificate_mode=section.as_str("certificate_mode", "determinant"),
        certificate_neighbourhood=section.as_int("certificate_neighbourhood", 1),
        use_constant_phase_residual=section.as_bool("use_constant_phase_residual", True),
        use_scale_free_estimand=section.as_bool("use_scale_free_estimand", True),
        use_tissue_mass_coupling=section.as_bool("use_tissue_mass_coupling", True),
        use_continuous_field=section.as_bool("use_continuous_field", True),
        use_identifiability_certificate=section.as_bool("use_identifiability_certificate", True),
        latent_dim=section.as_int("latent_dim", 96),
        encoder_width=section.as_int("encoder_width", 24),
        encoder_stages=tuple(int(value) for value in section.as_tuple("encoder_stages", 4)),
        decoder_width=section.as_int("decoder_width", 96),
        decoder_depth=section.as_int("decoder_depth", 3),
        encoding=section.as_str("encoding", "fourier"),
        bands=section.as_int("bands", 6),
        decoder_chunk=section.as_int("decoder_chunk", 1 << 16),
        classifier_hidden=section.as_int("classifier_hidden", 64),
        classifier_depth=section.as_int("classifier_depth", 2),
        dropout=section.as_float("dropout", 0.1),
        correction_width=section.as_int("correction_width", 16),
        correction_magnitude=section.as_float("correction_magnitude", 0.5),
        smoothness_weight=section.as_float("smoothness_weight", 1.0e-3),
    )


def synthetic_config(configuration: Configuration) -> SyntheticConfig:
    section = configuration.section("cohort")
    return SyntheticConfig(
        shape=section.as_shape("shape"),
        spacing_mm=(
            section.as_float_tuple("spacing_mm", 3)[0],
            section.as_float_tuple("spacing_mm", 3)[1],
            section.as_float_tuple("spacing_mm", 3)[2],
        ),
        parenchyma_modulus=section.as_float("parenchyma_modulus", 2.0),
        bulk_ratio=section.as_float("bulk_ratio", 10.0),
        traction=section.as_float("traction", 1.0),
        modulus_texture=section.as_float("modulus_texture", 0.04),
        lesion_margin=section.as_float("lesion_margin", 0.35),
        modulus_floor=section.as_float("modulus_floor", 1.0e-3),
        registration_floor=section.as_float("registration_floor", 0.05),
        registration_scale=section.as_float("registration_scale", 0.5),
        certificate_threshold=section.as_float("certificate_threshold", 0.5),
        certificate_ridge=section.as_float("certificate_ridge", 1.0e-8),
        certificate_mode=section.as_str("certificate_mode", "determinant"),
        boundary_shell=section.as_int("boundary_shell", 1),
        motion_sigma=section.as_float("motion_sigma", 0.0),
        follow_up_fraction=section.as_float("follow_up_fraction", 0.32),
        development_per_site=section.as_int("development_per_site", 240),
        seed=section.as_int("seed", 20260925),
        dtype=section.as_str("dtype", "float64"),
    )


def objective_weights(configuration: Configuration) -> ObjectiveWeights:
    section = configuration.section("objective")
    return ObjectiveWeights(
        classification=section.as_float("classification", 1.0),
        constant_phase_residual=section.as_float("constant_phase_residual", 0.3),
        tissue_mass_coupling=section.as_float("tissue_mass_coupling", 0.5),
        field_smoothness=section.as_float("field_smoothness", 1.0e-3),
        scale_free_gauge=section.as_float("scale_free_gauge", 1.0e-2),
        certified_only=section.as_bool("certified_only", False),
        label_smoothing=section.as_float("label_smoothing", 0.0),
    )


def training_spec(configuration: Configuration) -> TrainingSpec:
    section = configuration.section("training")
    epochs = section.as_int("epochs", 100)
    batch_size = section.as_int("batch_size", 4)
    grad_accumulation = section.as_int("grad_accumulation", 1)
    world_size = section.as_int("world_size", 1)
    dataset_size = section.as_int("dataset_size", 640)
    per_epoch = steps_per_epoch(dataset_size, batch_size, grad_accumulation)
    return TrainingSpec(
        epochs=epochs,
        batch_size=batch_size,
        grad_accumulation=grad_accumulation,
        world_size=world_size,
        seed=section.as_int("seed", 20260925),
        gradient_clip=section.as_float("gradient_clip", 1.0),
        log_every=section.as_int("log_every", 10),
        evaluation_every=section.as_int("evaluation_every", 1),
        early_stop_patience=section.as_int("early_stop_patience", 0),
        schedule=ScheduleSpec(
            base_lr=section.as_float("learning_rate", 3.0e-5),
            warmup_steps=section.as_int("warmup_steps", 500),
            total_steps=total_steps(epochs, per_epoch),
            minimum_factor=section.as_float("minimum_factor", 0.0),
            kind=section.as_str("schedule", "cosine"),
        ),
        optimizer=OptimiserSpec(
            name=section.as_str("optimizer", "adamw"),
            learning_rate=section.as_float("learning_rate", 3.0e-5),
            weight_decay=section.as_float("weight_decay", 1.0e-4),
            momentum=section.as_float("momentum", 0.9),
        ),
        precision=PrecisionSpec(kind=section.as_str("precision", "bf16")),
        averaging=AveragingSpec(
            decay=section.as_float("ema_decay", 0.999),
            warmup=section.as_int("ema_warmup", 0),
            enabled=section.as_bool("ema", True),
        ),
    )



def build_context(
    experiment: str, config_root: Path | None = None, overrides: tuple[str, ...] = ()
) -> ExperimentContext:
    configuration = resolve_experiment(experiment, config_root, overrides)
    cohort_config = synthetic_config(configuration)
    cohort = SyntheticCohort(cohort_config)
    probe = build_record(
        "probe-0000", cohort.entries[0].site, cohort.entries[0].outcome, cohort_config, 0
    )
    grid = spec_of([probe])
    return ExperimentContext(
        name=experiment,
        configuration=configuration,
        grid=grid,
        framework=framework_spec(configuration),
        weights=objective_weights(configuration),
        training=training_spec(configuration),
        cohort=cohort,
    )

def split_positions(context: ExperimentContext, split: Split, limit: int) -> list[int]:
    positions = [
        position
        for position, entry in enumerate(context.cohort.entries)
        if entry.site.split is split and entry.outcome.value in ("benign", "stage_IA")
    ]
    if limit > 0:
        positions = positions[:limit]
    return positions
