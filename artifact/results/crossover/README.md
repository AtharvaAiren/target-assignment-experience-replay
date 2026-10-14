# Dense crossover localization

This directory contains the compact, authenticated evidence used for the
outcome-informed crossover analysis. Each campaign retains its aggregate,
campaign binding, completion seal, and deterministic post-outcome analysis.
The 672 per-unit directories are omitted from this compact review artifact;
their inventories and digests were validated before these files were copied.

The primary results are discrete induced-support brackets:

- Permuted-MNIST versus conventional no replay: `26/31--27/31`.
- Task-aware Split CIFAR-100 versus conventional no replay:
  `30/31--31/31`.
- All-class Split CIFAR-100: no crossing on the tested grid; the contrast is
  still positive at full support.

Isotonic roots and bootstrap intervals in `ANALYSIS.json` are secondary,
model-based summaries. Censored profiles remain in the reported censoring
counts and extended-real interval endpoints.

From the artifact root, reproduce the analysis bundles with:

```bash
PYTHONPATH=. python3 -B experiments/gradient_reuse_crossover_analysis/analysis.py \
  --campaign pmnist \
  --lock configs/crossover_threshold_followup.json \
  --historical-aggregate results/corrected_reference_order/pmnist/AGGREGATE.json \
  --followup-aggregate results/crossover/pmnist/AGGREGATE.json \
  --followup-done results/crossover/pmnist/DONE.json \
  --followup-campaign-binding results/crossover/pmnist/CAMPAIGN.json \
  --output /tmp/pmnist-crossover-analysis.json

PYTHONPATH=. python3 -B experiments/gradient_reuse_crossover_analysis/analysis.py \
  --campaign cifar \
  --lock configs/crossover_threshold_followup.json \
  --historical-aggregate results/assignment_strength/aggregate.json \
  --followup-aggregate results/crossover/cifar/AGGREGATE.json \
  --followup-done results/crossover/cifar/DONE.json \
  --followup-campaign-binding results/crossover/cifar/CAMPAIGN.json \
  --output /tmp/cifar-crossover-analysis.json
```

Expected SHA-256 digests are
`39c4748c119b20d1566d259a6e970f0a02ef976a3f5c397bd9653945f3560641`
for PMNIST and
`888fc75489783622f1ceed7323beb7bb27784e86b0002833da0be8b3d4ce3127`
for CIFAR.

The public pre-outcome record, environment amendment, and sealed-source
manifest are under `provenance/crossover/`. The expanded crossover packages
and tests are included for inspection and ordinary use. The author-side source
archive is omitted from the anonymous release because its container metadata
contains machine-local owner strings; none of the reported values depends on
distributing that redundant container.
