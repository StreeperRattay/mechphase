#!/usr/bin/env bash
# Build the cohort index and its manifest. No network access is required.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"

python3 - <<'PY'
from mechphase.cohort.manifest import SyntheticCohort, expected_reported_counts
from mechphase.cohort.synthetic import SyntheticConfig
from mechphase.support.io import ensure_dir, write_json

root = ensure_dir(__import__("pathlib").Path("data"))
cohort = SyntheticCohort(SyntheticConfig())
manifest = cohort.manifest()
write_json(root / "cohort_index.json", {
    "records": [
        {
            "record_id": entry.record_id,
            "site": entry.site.name,
            "region": entry.site.region,
            "split": entry.site.split.value,
            "outcome": entry.outcome.value,
        }
        for entry in cohort.entries
    ],
    "expected": expected_reported_counts(),
    "constants": manifest.as_mapping(),
})
print(f"wrote {root / 'cohort_index.json'} with {len(cohort.entries)} records")
PY
