# How Target Assignment Shapes Experience Replay

Anonymous source, locked configurations, and compact evidence for the
accompanying TMLR manuscript. The study asks which benefits of experience
replay depend on preserving each stored input--target assignment. It compares
standard replay, target-reassigned replay, and current-only training; later
controls vary reassignment amount and structure, average over assignment
randomness, and match replay's scalar weight on current examples.

## Start here

- [Evidence map](docs/EVIDENCE.md): claims and their retained numerical records.
- [Reproduction guide](docs/REPRODUCE.md): checks, installation, and per-study
  execution requirements.
- [Data guide](docs/DATA.md): datasets, checkpoints, and expected input locations.
- [Provenance guide](docs/PROVENANCE.md): execution-time records and the public
  export boundary.

The artifact contains only experiments reported in the manuscript or its
appendix. Engineering pilots, failed attempts, server launchers, raw datasets,
checkpoints, original primary training traces, and duplicated run trees are
intentionally absent. The manuscript is supplied separately. Final manuscript
figure PDFs and their presentation scripts are outside this compact export.

Some internal module names, schema strings, and configuration fields contain
`iclr`. These are immutable historical identifiers from the campaign in which
the corresponding evidence was generated; they are not a venue claim and are
not renamed in this review release.

## Fast verification

The frozen evidence can be checked on a CPU-only machine without downloading
data or models:

```bash
python3 scripts/verify_results.py
python3 scripts/verify_paper_values.py
python3 scripts/audit_anonymity.py
```

The public anonymity audit checks machine paths, credentials, and release
clutter. A release maintainer should additionally pass author, account, host,
network, and linked-project identifiers through an external newline-delimited
private checklist that is never committed:

```bash
python3 scripts/audit_anonymity.py --private-token-file /path/outside/repo/private-tokens.txt
python3 scripts/sanitize_public_artifacts.py --check \
  --private-token-file /path/outside/repo/private-tokens.txt
```

Check the release manifest with `shasum -a 256 -c SHA256SUMS` on macOS or
`sha256sum -c SHA256SUMS` on Linux.

Or run the first three checks with `make verify audit`. The verifiers parse the
public JSON/JSONL evidence, check terminal records and expected study sizes,
and validate manuscript-facing invariants. They never rewrite results. They do
not retrain models, and for campaign families whose raw unit trees are not
distributed they do not reconstruct every aggregate from raw units. The exact
verification boundary is listed in [the reproduction guide](docs/REPRODUCE.md).

**Legacy Permuted-MNIST boundary.** A post-run source audit found that the
original Permuted-MNIST domain generator and replay sampling shared NumPy's
global random stream. No-replay and replay arms were therefore not guaranteed
to receive identical later domains. The frozen V_total, V_perm, and direct
cross-method PMNIST values remain available as historical arithmetic, but are
not interpreted as matched-stream effects. Within each method, the paired and
reassigned replay arms made matching buffer calls, so their V_pair comparison
remains in scope. The fresh corrected-reference follow-up now supplies
separate RNG-isolated PMNIST outcomes; it resolves the reference question for
that new ER campaign without retroactively validating the legacy arms. A
separate completed CIFAR campaign tests task-block and fully regrouped class
orders. The fresh exact-reference follow-up additionally matches replay's
scalar current-loss coefficient at every step on 12 new seeds. It removes the
remaining loss-weight ambiguity for that campaign while leaving replay
gradients and later trajectories deliberately unmatched.

## Included studies

| Study family | What it tests | Frozen evidence |
|---|---|---|
| Shallow streams and breadth | Local paired/permuted/reference signatures across regression, CNN, and recurrent settings | `results/scalar`, `results/reviewer`, `results/breadth` |
| Continual-learning panel | ER, GEM, A-GEM, MER, PCGrad and method boundaries on three benchmarks | `results/mechanisms`, `results/continual_primary` |
| Five-seed completion | Independent-seed replication and task-IL/held-out summaries | `results/primary_seed_completion` |
| Reviewer extension and language-model stress test | Off-floor/shared-state analyses, secondary endpoints, and Pythia feasibility | `results/reviewer_extension` |
| Reassignment dose | Partial target randomization over 12 independent seeds | `results/assignment_strength` |
| Dense crossover localization | Seven intermediate supports, with endpoints reused from completed parent studies | `results/crossover` |
| Conditional-mean control | Deterministic mean-matched targets against stochastic assignments | `results/mean_matched` |
| Assignment structure | Within-task versus across-task swaps at matched exposure | `results/task_structure`, `results/task_relative` |
| Shared-output and variance controls | Shared-label-space swap structure and fixed-mean variance ladder | `results/explanatory` |
| Corrected reference and class-order robustness | RNG-isolated PMNIST references and six locked CIFAR streams | `results/corrected_reference_order` |
| Exact stepwise coefficient match | Fresh-seed current-only controls that match replay's scalar current-loss coefficient at every update | `results/exact_reference` |
| Finite-horizon analysis | Exact 128-step linear propagation and assignment-noise term | `results/finite_horizon` |
| Controller appendix | Compact synthetic and Beijing controller summaries | `results/controller` |

See [the evidence map](docs/EVIDENCE.md) for the claim-to-file mapping and
[the reproduction guide](docs/REPRODUCE.md) for scientific entry points.

## Repository layout

```text
configs/                  paper experiment configurations
experiments/              study implementations
*_overlay/                minimal Mammoth-compatible interventions
src/chartfilm/            shared shallow-study components
results/                  compact, read-only paper evidence
provenance/               source and public-sanitization lineage
scripts/                  release verification utilities
tests/                    focused numerical and protocol tests
docs/                     data, evidence, provenance, and run guides
```

All new runs should be written outside this repository, for example in a
sibling directory named `../gradient-audit-reproduction/`. In particular, do
not write into the frozen `results/` tree. The release sanitizer was restricted
to replacing machine-local path strings. Author-retained original hashes,
public hashes, and replacement counts are recorded in
`provenance/SANITIZATION_MANIFEST.json`; the omitted original bytes cannot be
independently checked from this compact release.

The author-side sealed crossover source archive is intentionally not part of
the anonymous public release because its container metadata contains
machine-local owner strings. The shared tree later changed at three
manifest-bound paths. The retained historical manifest is therefore not
presented as a validation of the current tree unchanged. Instead,
`provenance/crossover/execution-source.patch` reconstructs those three exact
execution-time files in a disposable copy; the result verifier checks all 42
historical digests and fails closed on any additional drift. See
[`provenance/crossover/README.md`](provenance/crossover/README.md).

The two original mechanism source manifests likewise remain immutable
execution records. Anonymous public source renames one inert linked-project
isolation sentinel in exactly three files; it does not change unit selection,
training commands, optimization, metrics, or retained results. The historical
and public hashes are kept separate in
[`provenance/source/PUBLIC_SANITIZATION.json`](provenance/source/PUBLIC_SANITIZATION.json),
and the verifier rejects any unlisted difference rather than rewriting an
execution hash.

## Reproduction boundary

The original continual-learning campaign targets Mammoth commit
`e75a491c69fd729edeb01431afb753d9157d9a81` and uses the included
`mammoth_overlay/`. Later campaign-specific overlays are also included as
implementation references, but several historical follow-ups require source
closures or parent artifacts that are not redistributed. Raw third-party
datasets and pretrained model weights are not redistributed. The historical
language-model run requires the exact upstream snapshots named by repository
and revision in `configs/transformer.json`; the acquisition utility can create
a new local content-addressed copy, but the original cache is not included.
These limits do not affect CPU-only result verification. See
[the reproduction guide](docs/REPRODUCE.md) and [data guide](docs/DATA.md)
before attempting a live run. The sanitized Pythia compatibility implementation
and its historical-digest bridge are described in
[the Transformer correction note](docs/TRANSFORMER_CORRECTION.md).

This review-stage artifact is distributed under the limited terms in
[LICENSE](LICENSE), not under an open-source license. Third-party components
remain subject to their own terms; see
[`mammoth_overlay/NOTICE.md`](mammoth_overlay/NOTICE.md). Verbatim copies of
the pinned Mammoth MIT license and GEM CC BY-NC 4.0 license are bundled as
[`third_party_licenses/MAMMOTH-MIT.txt`](third_party_licenses/MAMMOTH-MIT.txt)
and
[`third_party_licenses/GEM-CC-BY-NC-4.0.txt`](third_party_licenses/GEM-CC-BY-NC-4.0.txt);
third-party portions remain governed by those terms rather than this
repository's review-only notice.
