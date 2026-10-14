# RTX seven-day mechanism-audit profile

## Prespecified compute-budget amendment

This profile was locked before its scientific run to fit the ICLR extension
campaign within a seven-day wall-clock budget on one 32 GiB RTX GPU. It is a
separate evidence profile; it does not modify or supersede the original
five-seed mechanism lock in `configs/mechanisms.json`.

The sole scientific-design change is the number of independent run seeds:

- original full profile: `3401, 3402, 3403, 3404, 3405`;
- RTX seven-day profile: `3401, 3402, 3403`.

Everything else is unchanged. The profile retains Split-CIFAR-100,
Split-miniImageNet, and Permuted-MNIST; all eight methods; the paired arm and
all four nested target-permutation replicas for every replay method; all
method hyperparameters; and every evaluation endpoint. It therefore contains
288 paper units: for each of three benchmarks and three seeds, two reference
controls plus six replay methods with one paired and four permuted arms.

The synthetic and Beijing controller experiments and the complete Pythia-160M
experiment use their original locked configurations without reduced seeds or
endpoints. Four deterministic mechanism shards are the default execution
layout and run one at a time on the single RTX GPU. Sharding changes failure
isolation and resumability only, not unit identity or inference.

## Interpretation and safeguards

Three independent seeds yield wider Student-t intervals and lower power than
the original five-seed plan. The four permutation replicas remain nested Monte
Carlo draws and are never treated as independent run seeds. Results from this
profile must be reported as the RTX seven-day profile and must not be pooled
with partial or complete outputs from the 480-unit profile.

The amended configuration records the original lock's byte-level and canonical
SHA-256 digests. Validation reconstructs the original lock and fails if any
field other than `runtime.seeds` changed. The dedicated launcher requires a
fresh root on first launch and accepts only the exact immutable campaign
profile on resume. It authenticates all four deterministic shards and
materializes a 288-unit aggregate
before the Transformer stage can begin. The unchanged Pythia campaign has
three immutable attempt slots, so a final-stage interruption preserves prior
controller and mechanism evidence while retrying only the Transformer stage.
Completion is recorded as `RTX_7DAY_PROFILE_DONE.json`; this reduced profile
never writes or claims the stock full campaign's completion marker.
