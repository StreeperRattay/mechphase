"""Shared fixtures: small volumes, a tiny framework and reusable seeds."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mechphase.cohort.batching import collate, spec_of  # noqa: E402
from mechphase.cohort.schema import SiteSpec  # noqa: E402
from mechphase.cohort.synthetic import SyntheticConfig, SyntheticRecord, build_record  # noqa: E402
from mechphase.estimators.framework import FrameworkSpec, MechPhaseModel  # noqa: E402
from mechphase.loop.rng import set_seed  # noqa: E402
from mechphase.support.types import GridSpec, Outcome, Split  # noqa: E402

SEED = 20260925
SMALL_SHAPE = (16, 12, 12)
PROSPECTIVE_SITE = SiteSpec(
    name="Site D", region="Region III", count=486, split=Split.PROSPECTIVE
)
DEVELOPMENT_SITE = SiteSpec(
    name="Site A", region="Region I", count=214, split=Split.DEVELOPMENT
)


@pytest.fixture(autouse=True)
def _deterministic_seed() -> Iterator[None]:
    set_seed(SEED)
    yield


@pytest.fixture
def grid() -> GridSpec:
    return GridSpec(shape=SMALL_SHAPE, spacing_mm=(1.0, 1.0, 1.0))


@pytest.fixture
def synthetic_config() -> SyntheticConfig:
    return SyntheticConfig(shape=SMALL_SHAPE)


@pytest.fixture
def framework_spec() -> FrameworkSpec:
    return FrameworkSpec(
        encoder_width=8,
        latent_dim=32,
        decoder_width=32,
        decoder_depth=2,
        correction_width=8,
        decoder_chunk=1 << 14,
        certificate_threshold=0.0,
    )


def make_record(
    index: int,
    outcome: Outcome = Outcome.STAGE_IA,
    config: SyntheticConfig | None = None,
) -> SyntheticRecord:
    resolved = SyntheticConfig(shape=SMALL_SHAPE) if config is None else config
    return build_record(
        f"test-{index:04d}", PROSPECTIVE_SITE, outcome, resolved, index
    )


@pytest.fixture
def record() -> SyntheticRecord:
    return make_record(0)


@pytest.fixture
def records() -> list[SyntheticRecord]:
    return [make_record(index, Outcome.STAGE_IA if index % 2 == 0 else Outcome.BENIGN) for index in range(4)]


@pytest.fixture
def batch(records: list[SyntheticRecord]) -> dict[str, torch.Tensor]:
    return collate(records)


@pytest.fixture
def model(records: list[SyntheticRecord], framework_spec: FrameworkSpec) -> MechPhaseModel:
    return MechPhaseModel(spec_of(records), framework_spec)
