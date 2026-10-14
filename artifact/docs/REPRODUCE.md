# Reproduction and verification guide

This release supports two different activities:

1. **Evidence verification** reads the frozen public result files and checks
   inventories, terminal status, bindings, and manuscript-facing invariants.
2. **Live reproduction** executes code against newly acquired inputs. It needs
   external data, and some campaigns additionally need a pristine checkout of
   Mammoth or exact upstream model snapshots.

The two activities are not interchangeable. A module exposing `--help` does
not by itself mean that the complete historical execution closure is included.

## 1. Verify the frozen evidence

```bash
python3 scripts/verify_results.py
python3 scripts/verify_paper_values.py
```

This is the recommended reviewer path and is CPU-only. Add the anonymity and
release-tree checks with:

```bash
python3 scripts/audit_anonymity.py
shasum -a 256 -c SHA256SUMS       # macOS
# sha256sum -c SHA256SUMS         # Linux
```

The distributed audit intentionally contains no author, account, or host-name
checklist. Release maintainers should add their private terms at invocation
time with `--private-token` or an external `--private-token-file`; invented
tokens exercise this interface in the test suite.

`verify_results.py` parses the public result files, checks selected terminal
statuses, and checks the explicitly enumerated unit counts.
`verify_paper_values.py` checks reported study counts and selected numerical
invariants; it also round-trips the compact continual-primary endpoint
projection against the frozen aggregate from which that projection was made.
Neither command retrains a model, and neither is a universal schema,
raw-unit-to-aggregate, or raw-to-table verifier. Run the test suite for protocol
and numerical unit tests, and check `SHA256SUMS` separately for release-file
integrity. `docs/EVIDENCE.md` identifies the authoritative file and residual
verification boundary for each reported family.

The original mechanism manifests remain execution-time records. The public
locks use a neutral name for an inert linked-project isolation sentinel, while
the frozen results continue to cite the historical hashes. The explicit
three-file mapping is `provenance/source/PUBLIC_SANITIZATION.json`; the result
verifier checks both identities and rejects any unlisted source drift. Exact
reconstruction of the three private historical strings requires a
maintainer-only token that is intentionally outside this anonymous release.

## 2. Install the local code

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
export PYTHONPATH="$PWD:$PWD/src"
```

Install the optional `vision` or `transformer` dependency group only for the
corresponding campaigns. The pinned Mammoth code imports Weights & Biases even
when online logging is disabled, so Mammoth-based runs additionally require:

```bash
python -m pip install 'wandb>=0.16'
```

This is a runtime prerequisite, not permission to enable network logging.
Run Mammoth with its offline/disabled logging mode as required by the selected
campaign.

## 3. Capability matrix

| Campaign family | Public evidence verification | Live execution from this release | Required external inputs |
|---|---|---|---|
| Shallow scalar streams and reviewer diagnostics | Included compact results pass the selected public checks | Supported by the shipped source freezes | UCI archives; the loaders use the fixed cache described in `docs/DATA.md` |
| CNN/GRU breadth | Included compact results pass the selected public checks | Supported by `experiments.fmmt_iclr_breadth`; `iclr` is a historical identifier | CIFAR-100 at the fixed repository-relative cache and the UCI Appliances archive |
| Original continual-learning mechanisms | Included compact aggregate passes the selected checks; the raw unit trees are not distributed | Supported after preparing the pinned Mammoth checkout, installing `mammoth_overlay/`, and supplying locked data/runtime manifests. The legacy Permuted-MNIST reference-arm limitation below remains part of that historical protocol | CIFAR-100, MNIST, local miniImageNet, CUDA, Mammoth commit `e75a491c69fd729edeb01431afb753d9157d9a81` |
| Five-seed completion | Included compact aggregate passes the selected checks | **Implementation/reference only in this release.** The exact parent E2 export and E5 unit tree required by the queue are not distributed | Authenticated parent E2 export, E5 unit tree, pinned Mammoth, data/runtime manifests, CUDA |
| Original Pythia audit | Included compact result passes the selected checks | **Implementation/reference only in this release.** The original result schema does not bind the later portable correction bytes, so the public wrapper is not represented as an authenticated recreation of the historical run | Exact Hugging Face revisions in `configs/transformer.json`, local asset manifest, suitable CUDA runtime, historical correction closure |
| Synthetic/natural controllers | Included compact results pass the selected checks | Supported; full runs require captured runtime manifests, and the natural controller also requires a locked Beijing archive | None for synthetic smoke; Beijing archive for the natural campaign |
| Finite-horizon Beijing analysis | Included compact result and 60 unit records pass the selected checks | Supported by the shipped lock and source manifest | Exact Beijing archive with SHA-256 in `configs/finite_horizon_beijing.json` |
| Assignment-strength, mean-matched, task-structure, task-relative, and explanatory follow-ups | Included compact aggregates pass the selected checks | **Implementation/reference only in this release.** Historical source builders include deployment-only files that were intentionally excluded, so a paper-eligible historical closure cannot be regenerated here | Pinned Mammoth, data/runtime manifests, omitted historical deployment closure and, for dependent campaigns, authenticated parent artifacts |
| Reviewer-extension continual-learning and amended Pythia campaigns | Included compact results pass the selected checks; 156 reviewer-extension records are counted | **Verification/reference only in this release.** A portable implementation of three documented correction rules is bundled, but its equivalence to the unavailable historical patch bytes is not established, and the historical parent/launch closure is incomplete | Authenticated parent campaign, external data/assets, historical launch closure, and pinned runtimes |
| Fresh corrected-reference/order audits | Completed compact aggregates, bindings, and seals are included for 228 PMNIST and 504 CIFAR units; raw unit trees are not distributed | Runnable as two separately locked, fail-closed campaigns; a new execution must use fresh external output roots and the disclosed source amendments | Pristine pinned Mammoth, locked data/CUDA environment manifests, source-closure manifest, and separate external output roots |
| Exact stepwise coefficient-matched reference | Completed compact aggregates, bindings, seals, and a combined seal are included for 240 PMNIST and 576 CIFAR units; raw unit trees are not distributed | Runnable from the locked exact-reference configurations and retained source closure; a new execution must use fresh external output roots | Pristine pinned Mammoth, locked data/CUDA environment manifests, source manifest, and CUDA |
| Dense crossover localization | Two completed 336-unit aggregates, bindings, completion records, and deterministic analysis bundles are included; raw unit trees are not distributed | The analysis is reproducible from the compact public evidence. Fresh training requires the pinned Mammoth, parent aggregates, data/runtime manifests, and new external output roots | Parent aggregates, pristine pinned Mammoth, locked data/CUDA environment manifests, and CUDA |

The implementation/reference rows are useful for reading the intervention,
running unit tests, and designing a fresh independently locked replication.
They should not be represented as commands that recreate the authenticated
historical run.

The environment specifications in `pyproject.toml` and `environment.yml` are
installation guides, not byte-exact solver locks. Frozen result records retain
the observed package and accelerator versions for the principal campaigns.
Accordingly, “supported” above means that the public scientific entry point and
its input contract are present; exact numerical replay additionally requires
reconstructing a compatible environment from the recorded manifest.

The portable Transformer correction implementation and its explicit
author-recorded historical-digest bridge are documented in
[TRANSFORMER_CORRECTION.md](TRANSFORMER_CORRECTION.md).
The corrected campaign protocol is documented in
[corrected_reference_order_runbook.md](corrected_reference_order_runbook.md),
and the bounded execution amendments are documented in the two
`POST_OUTCOME_*_AMENDMENT.md` files.

The current executable corrected-campaign tree is later than both historical
execution manifests. Two manifest-bound documentation files were updated after
completion, the shared intervention later acquired the exact-reference
`drop_last` audit field, and the earlier PMNIST snapshot also predates the
CIFAR-only multiline-log parser amendment. The verifier checks the exact
allowed difference set for each campaign. For a new run, generate a fresh
source manifest from the current tree rather than reusing either historical
execution manifest.

The crossover manifest likewise binds an earlier shared-tree state. To inspect
that exact source without changing this release, copy the repository to a
temporary directory and run:

```bash
patch -p1 < provenance/crossover/execution-source.patch
python3 -c "from pathlib import Path; from experiments.gradient_reuse_crossover.source import validate_source_manifest; validate_source_manifest(Path('provenance/crossover/crossover-source-manifest.json')); print('CROSSOVER SOURCE: PASS')"
```

The release verifier performs this reconstruction in a temporary directory and
requires all 42 historical digests to match. Fresh training from the current
tree requires a new source identity; do not overwrite the historical manifest.

The exact-reference protocol and audit boundary are documented in
[`exact_reference_followup_lock.md`](exact_reference_followup_lock.md) and
[`POST_FAILURE_DROP_LAST_AUDIT_AMENDMENT.md`](POST_FAILURE_DROP_LAST_AUDIT_AMENDMENT.md).
The retained execution-time source manifest matches the current listed source
files exactly. Public verification authenticates the compact aggregates and
seals but cannot replay the omitted raw unit-to-aggregate reduction.

## 4. Live entry points with a documented public input contract

Each package exposes `--help`; configurations under `configs/` contain the
reported study axes. Representative supported entry points are:

```bash
PYTHONPATH=.:src python -m experiments.fmmt_workshop.run --help
PYTHONPATH=.:src python -m experiments.fmmt_reviewer.run --help
PYTHONPATH=.:src python -m experiments.fmmt_iclr_breadth.run --help
PYTHONPATH=.:src python -m experiments.gradient_reuse_mechanisms.run --help
PYTHONPATH=.:src python -m experiments.gradient_reuse_controller.run --help
PYTHONPATH=.:src python -m experiments.gradient_reuse_controller.natural --help
PYTHONPATH=.:src python -m experiments.gradient_reuse_finite_horizon.study --help
PYTHONPATH=.:src python -m experiments.gradient_reuse_exact_reference.queue --help
```

For the original continual-learning campaign:

1. Clone Mammoth separately and checkout commit
   `e75a491c69fd729edeb01431afb753d9157d9a81`.
2. Verify that checkout is pristine before installing the files from
   `mammoth_overlay/`; do not layer the later campaign-specific overlays onto
   the original mechanism run.
3. Acquire and freeze the data layout in `docs/DATA.md`.
4. Capture the runtime/environment manifest required by the runner.
5. Use `experiments.gradient_reuse_mechanisms.run --help` to select the full
   campaign or a unit/shard. Write output outside this repository.

The follow-up overlay directories (`assignment_strength_overlay/`,
`mean_matched_overlay/`, `task_structure_overlay/`,
`explanatory_followup_overlay/`, and `reviewer_extension_overlay/`) correspond
to different locked campaigns. They are not cumulative patches and must not be
mixed. Their historical launch closures are not complete in this release.

## 5. Safe output policy

Never overwrite `results/`, its manifests, or any file named by
`SHA256SUMS`. Use an external path, for example:

```bash
mkdir -p ../gradient-audit-reproduction
```

Smoke outputs are engineering checks and are not substitutes for the locked
scientific tiers.

After the final release tree has been audited and `SHA256SUMS` has been
regenerated, build the submission archive at a new path outside the repository:

```bash
python3 scripts/build_release_archive.py \
  --output ../gradient-audit-release.zip
```

The builder treats `SHA256SUMS` as the sole payload allowlist, rejects symbolic
or noncanonical members, verifies each payload before writing, reopens and
checks the completed ZIP, and refuses to overwrite an existing output.
