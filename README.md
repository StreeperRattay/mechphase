# Lung Mechanical Property Mapping by Physics-Constrained Deep Learning for Early Adenocarcinoma Detection

This package is the code release for the constant-phase mechanical property mapping framework
described in *Lung Mechanical Property Mapping by Physics-Constrained Deep Learning for Early
Adenocarcinoma Detection*. It recovers two properties of lung parenchyma, the lesion-relative modulus
ratio `rho` and the dimensionless loss tangent `eta`, from paired inspiratory and
expiratory computed tomography, without contrast, external loading or dedicated elastography.
The release contains the five components of the framework (the constant-phase residual, the
scale-free estimand, tissue-mass coupling, the continuous implicit field and the identifiability
certificate), the encoder, the comparators of the diagnostic comparison, the statistical layer,
and the robustness, transfer, reader-study and in-vitro analyses.

## Project context

| Field | Value | Source in the manuscript |
| --- | --- | --- |
| Domain | Computational imaging: paired-CT mechanical property mapping for pulmonary adenocarcinoma | Sec. 1, Sec. 4.2 |
| Framework | PyTorch 2.x with plain `torch.nn`, `torch.autograd` and finite differences on the voxel grid | Sec. 4.3, Sec. 4.4 |
| Venue | npj Digital Medicine | journal formatting and structured supplementary information |
| Estimand | The scale-free pair `(rho, eta)` with `rho = G_lesion / G_parenchyma` and `eta = G'' / G'` | Sec. 2.2, Proposition 1, Assumptions (A1)-(A3) |
| Primary cohorts | Four private anonymised clinical cohorts: development, prospective, external validation, calibration | Sec. 4.1 |
| Public auxiliary collections | NLST, LIDC-IDRI, LUNA16, Learn2Reg CT lung registration | Table S6 |
| Compute target | 4 x NVIDIA A100 80 GB for 38 h including the hyperparameter search; 1 x A100 40 GB for inference at 6.2 s per examination; 41 kg CO2e | Table S8 |
| Reported primary result | AUC 0.940 (0.922-0.958), sensitivity 89.3 %, specificity 92.1 %, abstention 12.7 % | Table 2 |
| Pre-specified constants | negligible-difference threshold 0.03; primary criterion AUC >= 0.900 and sensitivity >= 85 %; minimal clinical difference 5.0 points; decision threshold 0.20; stability cut-off 0.40-0.50 with an AUC drift of at most 0.01 | Sec. 4.7 |

### What the manuscript fixes and what this release fixes

Values the manuscript prints are used verbatim: the arm composition, the site counts of the
prospective and external arms, the pre-specified statistical constants, the accelerator and
wall-clock accounting, and the phenotype medians and interquartile ranges of Table 1.

Values the manuscript does not print are exposed as engineering defaults in
`configs/` and listed under `provenance.engineering_defaults` in every experiment file:
the optimiser, schedule, learning rate, warm-up, weight decay, precision, batch size, epoch
budget, gradient accumulation and the relative weights of the objective terms; the certificate
threshold and ridge; the voxel spacing; the per-site counts of the development arm; and the
coefficient vectors of the clinical instruments.

### Outcome of the verification pass

`verification_report.json` records one entry per check with the value that was actually
observed. The closed-form checks compare the implementation with hand-derived relations, the
statistical checks compare it with SciPy and scikit-learn or with a brute-force recomputation
written inside the check itself, and the execution checks run the data path, the forward pass,
the objective, the backward pass, a parameter update, a checkpoint round trip, a single-batch
overfit and a minimal training loop. Cohort-level endpoints are not recomputed because the
clinical cohorts are private; that is recorded as `NOT_RUN` rather than as a result. The
reachability of the recorded collection links is likewise `NOT_RUN` unless the pass is run with
`--probe-links`, because whether a route answers is a property of the current network rather
than of the release. The
container image is recorded as `BLOCKED` because this verification host has no container
runtime, and the frozen encoder is recorded as `BLOCKED` because its weights are not
redistributable. `claim_to_code.json` carries the paper-claim mapping together with every
recorded deviation.

## Repository layout

```
configs/
  model/mechphase.yaml            constitutive constants, framework components, certificate
  data/cohort.yaml                generated-cohort geometry and access terms
  train/default.yaml              optimiser, schedule, objective weights, runtime
  train/_smoke.yaml               two-epoch configuration used only by the test suite
  experiment/main.yaml            the primary configuration
  experiment/ablation_*.yaml      one file per reported ablation of Table 3
  experiment/control_*.yaml       morphology-only and shuffled-map controls
  experiment/supplementary_*.yaml seed sweep, perturbations, transfer, reference, sweeps
src/mechphase/
  constitutive/                   constant-phase law, stress assembly, scale equivariance
  operators/                      grid operators, coordinate autograd, Jacobian audits, smoothing
  inverse/                        equilibrium residual, boundary traction, mass coupling, certificate
  fields/                         coordinate encodings, implicit field, decoder, regularisation
  imaging/                        volumes, registration surrogate, regions of interest, kernels
  cohort/                         schema, generator, manufactured state, partition, reference
  estimators/                     encoder, implicit field assembly, comparators, radiomics, clinical
  objectives/                     component terms and the weighted assembly
  statistics/                     discrimination, DeLong, multiplicity, resampling, decision curves
  analysis/                       phenotype, ablations, perturbations, transfer, readers, in vitro
  loop/                           seeding, schedule, optimiser, precision, checkpoints, session
  harness/                        fit, score, audit, claims and artefact writers
scripts/                          launch_train.sh, launch_eval.sh, prepare_data.sh
tests/                            unit, shape, integration, overfit, training-loop and driver checks
```

## Installation

With pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,plots]'
```

With conda:

```bash
conda env create -f environment.yml
conda activate mechphase
```

With Docker:

```bash
docker build -t mechphase .
docker run --rm mechphase --help
```

## Data

No clinical data ship with this repository. The four clinical cohorts are held by the
originating sites and are available only from the data holders under a data-use agreement.
`src/mechphase/cohort/synthetic.py` builds a schema-compatible stand-in cohort with the same
per-record fields, the same geometry contract and a closed-form mechanical ground truth, so
every mechanism in the release is executable without the private data. The stand-in is a
substitute for execution and verification only; every cohort-level number in the manuscript
must be read from the manuscript.

The public auxiliary collections listed below are used for encoder pretraining and for
deformation sanity checks only. Each entry records the version, the licence and the access
route as reported in the manuscript's own inventory. `dataset_urls.txt` holds that recorded
inventory, which is the four routes below. Whether a route answers from a given host is a
property of the network rather than of the release, and it therefore stays out of every file
the integrity manifest digests: the `public_dataset_links` check is `NOT_RUN` by default and
`harness.audit --probe-links` performs the reachability check on demand. When probed during
this release, the two Zenodo records answered from the verification host and the two TCIA
collection pages did not, which is an egress limitation of that host rather than evidence
that the collections are retired.

| Collection | Version | Licence | Access | Role |
| --- | --- | --- | --- | --- |
| NLST imaging collection, TCIA | Version 3 | CC BY 4.0, no application | https://www.cancerimagingarchive.net/collection/nlst/ | encoder pretraining only |
| LIDC-IDRI, TCIA | Version 4 | CC BY 3.0, no application | https://www.cancerimagingarchive.net/collection/lidc-idri/ | nodule and segmentation pretraining only |
| LUNA16 release, Zenodo record 3723295 | v3 | CC BY 4.0, no application | https://zenodo.org/records/3723295 | nodule-detection pretraining only |
| Learn2Reg CT lung registration data, Zenodo record 3835682 | v1 | CC BY 4.0, no application | https://zenodo.org/records/3835682 | registration and deformation sanity check |

Preparing the stand-in cohort (no network access required):

```bash
bash scripts/prepare_data.sh
```

The command writes the cohort index and its manifest under `data/`. The generated volumes are
built on demand rather than materialised: `SyntheticCohort.__getitem__` derives each record
deterministically from its index and the configured seed, so the cohort reproduces exactly
without a large on-disk footprint. Removing the cache and rebuilding returns an identical
index; `tests/test_cohort.py` asserts that property.

## Training

One command per reported configuration:

```bash
python -m mechphase.harness.fit --experiment main
python -m mechphase.harness.fit --experiment ablation_without_cpr
python -m mechphase.harness.fit --experiment ablation_without_sir
python -m mechphase.harness.fit --experiment ablation_without_cpr_and_sir
python -m mechphase.harness.fit --experiment ablation_without_tmc
python -m mechphase.harness.fit --experiment ablation_without_cif
python -m mechphase.harness.fit --experiment ablation_without_fic
python -m mechphase.harness.fit --experiment control_morphology_only
python -m mechphase.harness.fit --experiment control_shuffled_map
python -m mechphase.harness.fit --experiment supplementary_seed_variance
```

`--override training.epochs=4` and any other dotted key can be appended to any command. The
five-run seed sweep of Table S3 uses the seeds `[23, 340, 657, 974, 1291]` recorded in the
supplementary configuration; the runs differ only in the seed.

Every run writes `runs/<experiment>.pt` and `runs/<experiment>_report.txt`. The checkpoint
carries the optimiser state, the step and epoch counters, the seed and the random-number
state, and is written atomically with a temporary file that is renamed into place.

## Evaluation

```bash
python -m mechphase.harness.score --experiment main --limit 64
python -m mechphase.harness.audit
```

`harness.score` writes `runs/<experiment>_score.txt` with the discrimination, the certificate
composition, the decision curve at the reported threshold and the stratified increment over the
configured cohort. Because the clinical cohorts are private, the reported analysis-set numbers
are not recomputed here and the score report says so in its own text; the stand-in cohort is
not the analysis set.

`harness.audit` runs the full verification pass described above and writes
`claim_to_code.json`, `verification_report.json`, `verification_summary.txt`,
`dataset_urls.txt` and `integrity_manifest.json` in that order, then re-hashes the live tree and
diffs it against the manifest. The command exits non-zero if the manifest and the tree disagree.
Every artefact it writes is a pure function of the repository, so two consecutive runs produce
byte-identical output and a fresh clone that runs the pass leaves the tree clean. Add
`--probe-links` to probe the recorded collection links from the current host, and `--skip-tools`
to omit the ruff, mypy and pytest checks.

## Compute budget

| Item | Value | Source |
| --- | --- | --- |
| Training accelerators | 4 x NVIDIA A100 80 GB, same generation | Table S8 |
| Training wall-clock | 38 h, including the hyperparameter search | Table S8 |
| Inference accelerator | 1 x NVIDIA A100 40 GB | Table S8 |
| Inference latency | 6.2 s per examination | Table S8 |
| Estimated emissions | 41 kg CO2e using the ML CO2 Impact convention | Table S8 |
| Effective batch size | batch size 4 x gradient accumulation 1 x 4 ranks | configuration default |
| Peak disk for the cohort index | under 5 MB; volumes are generated on demand | configuration default |

The verification pass in this release ran on a single CPU host. No accelerator was used to
produce any artefact in the repository, and the wall-clock figures above are the manuscript's
reported budget rather than a measurement made here.

## Third-party components

`NOTICE` names every third-party dataset and package with its real licence and source URL, and
`LICENSE` carries the release licence. No third-party data are redistributed; only the access
routes and terms are recorded.
