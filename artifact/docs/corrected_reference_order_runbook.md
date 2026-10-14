# Corrected reference and order campaigns

These are fresh, prospective campaigns. Their scientific designs and analysis
locks were sealed before any corrected-campaign outcome was inspected. The
execution source was recorded at each launch. Two bounded post-outcome
engineering amendments required clean restarts and are disclosed separately:
the adapter-seed repair applies to both completed campaigns, while the later
multiline-log parser repair applies only to the restarted CIFAR campaign. The
campaigns subsequently completed, and compact aggregates, bindings, and seals
are now included under `results/corrected_reference_order/`. They do not
modify, replace, or reinterpret the earlier frozen evidence under `results/`;
the fresh PMNIST record resolves the reference question only for its own ER
campaign.

## Locked inventories

The campaigns are deliberately independent and require different output roots.

| Campaign | Lock | Complete inventory | Completion rule |
|---|---|---:|---|
| Permuted-MNIST corrected reference | `configs/corrected_reference_order_audit.json` | 228 | All 12 seeds (3801--3812) and all 19 arms/strength-replicas per seed |
| Split CIFAR-100 order sensitivity | `configs/corrected_cifar_order_audit.json` | 504 | All 12 seeds, six locked orders, and seven arms/replicas per seed/order |

The PMNIST grid contains standard no-replay SGD at 0.1, a nominal
learning-rate-rescaled no-replay control at 0.08, paired ER at lambda 0, and
four nested assignment replicas at each of lambda 0.25, 0.5, 0.75, and 1.0.
Its outcome-blind queue
puts the 132 standard/fixed/lambda-0/lambda-0.75/lambda-1 units first, but that
prefix is not a scientific aggregate or a completed campaign. All 228 units
remain mandatory.

PMNIST first averages the four nested assignment replicas within each seed and
nonzero lambda. Its prospectively locked seven-test Holm family is: the
within-seed dose slope over lambda 0, 0.25, 0.5, 0.75, and 1 (using paired ER
at zero); paired-minus-standard; paired-minus-fixed; and replica-mean
reassigned-minus-standard and reassigned-minus-fixed at lambda 0.75 and 1.
Positive slopes mean accuracy rises with reassignment strength; every other
sign convention is written explicitly in the lock. At all nonzero lambdas the
aggregate also reports paired-minus-reassigned and reassigned-minus each
no-replay reference as descriptive pointwise summaries.

The CIFAR grid crosses three locked task-block orders and three locked
full-class orders with standard no-replay SGD at 0.03, a nominal
learning-rate-rescaled no-replay control at 0.015, paired ER at lambda 0, and
four nested lambda-1 assignment replicas. Its six confirmatory tests are the
two order families by
`V_pair`, paired-minus-standard, and paired-minus-fixed. Each contrast is first
averaged across the three orders within run seed and family; Holm adjustment is
then applied to the six seed-level tests. Orders and assignment replicas never
inflate the independent-seed count.

The rescaled controls multiply the no-replay learning rate by the nominal
full-minibatch coefficient on the current gradient (0.8 for PMNIST and 0.5 for
CIFAR). They are not exact dynamic reweightings: the ER coefficient differs
when its buffer is empty and on a final partial current minibatch. These arms
therefore diagnose the fixed learning-rate explanation without claiming to
hold every realized update component identical.

Replay-arm equality is checked fail-closed using fingerprints of the
prespecified replay events sampled at the locked 100-observation audit
interval. This check covers those audited events, not every replay event.

## Preflight and source seal

Use a pristine checkout of Mammoth commit
`e75a491c69fd729edeb01431afb753d9157d9a81`, the existing assignment-strength
intervention, the locked dataset manifest, and a captured CUDA environment
manifest. The runner verifies the commit and license, rejects dirty Mammoth
trees, validates every dataset byte, checks the environment and Python binary,
requires CUDA device 0, and checks free disk/GPU memory. PMNIST task banks use
an isolated local NumPy generator and are digest-checked across every arm;
CIFAR class streams, inverse maps, and all 20 observed task sets are checked at
runtime. The CIFAR overlay also converts the pinned parser's authenticated
custom-order Python list to the NumPy array representation required by the
pinned loader; this is a value-preserving, RNG-free compatibility repair.

For a new reproduction, create a new source closure from the current public
tree at a new external path. The historical execution manifests under
`provenance/corrected_reference_order/` document the completed runs and must
not be reused as manifests for modified documentation or source bytes:

```bash
python scripts/build_corrected_reference_order_manifest.py \
  --output /absolute/external/locks/corrected-source-manifest.json
python scripts/build_corrected_reference_order_manifest.py \
  --output /absolute/external/locks/corrected-source-manifest.json --check
```

Preview either exact queue without a GPU or any write:

```bash
python -m experiments.gradient_reuse_corrected_reference_order.queue \
  --campaign pmnist --lock configs/corrected_reference_order_audit.json
python -m experiments.gradient_reuse_corrected_reference_order.queue \
  --campaign cifar --lock configs/corrected_cifar_order_audit.json
```

## Execution

Training is disabled unless `--execute` is present. Output must be an absolute
external directory that does not overlap this repository; never use `results/`.
Run PMNIST and CIFAR in separate empty roots:

```bash
python -m experiments.gradient_reuse_corrected_reference_order.queue \
  --campaign pmnist --lock configs/corrected_reference_order_audit.json \
  --execute \
  --source-manifest /absolute/external/locks/corrected-source-manifest.json \
  --mammoth-root /absolute/external/Mammoth \
  --data-root /absolute/external/data \
  --data-manifest /absolute/external/locks/data-manifest.json \
  --environment-manifest /absolute/external/locks/environment-manifest.json \
  --python /absolute/external/venv/bin/python \
  --output-root /absolute/external/runs/corrected-pmnist-228

python -m experiments.gradient_reuse_corrected_reference_order.queue \
  --campaign cifar --lock configs/corrected_cifar_order_audit.json \
  --execute \
  --source-manifest /absolute/external/locks/corrected-source-manifest.json \
  --mammoth-root /absolute/external/Mammoth \
  --data-root /absolute/external/data \
  --data-manifest /absolute/external/locks/data-manifest.json \
  --environment-manifest /absolute/external/locks/environment-manifest.json \
  --python /absolute/external/venv/bin/python \
  --output-root /absolute/external/runs/corrected-cifar-504
```

Failed attempts remain in their unit directory and successful unit seals are
validated before resume. `AGGREGATE.json` and the campaign `DONE.json` are
written only after exact, duplicate-free full coverage (228 or 504); partial
scientific aggregation is rejected.

Run the implementation checks without launching training:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/test_corrected_reference_order_bank.py \
  tests/test_corrected_reference_order_protocol.py \
  tests/test_corrected_reference_order_aggregate.py
```
