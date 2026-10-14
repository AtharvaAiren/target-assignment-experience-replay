# Dense reassignment crossover analysis

This post-run module analyzes the outcome-informed dense-strength follow-up. It
does not alter any historical result and must not be run on a partial campaign.

## Locked estimand and grid

The estimand is the same-seed final-average-accuracy contrast

`replica-mean reassigned ER - conventional no replay`.

For Permuted-MNIST, the complete analysis is repeated against the authenticated
fixed-current no-replay control as a separately labelled sensitivity. Its
crossing is never pooled with the conventional-control primary crossing.

For replay minibatches of 32, the legacy 7,500 and 10,000 basis-point requests
induce support strengths 23/31 and 31/31. The seven new requests 7,742, 8,065,
8,387, 8,710, 9,032, 9,355, and 9,677 basis points induce every intermediate
support strength 24/31 through 30/31. Requested strength, induced support
strength, and observed changed-label fraction are retained separately.

Assignment replicas are nested Monte Carlo draws. They are averaged within
seed and strength before inference and are never counted as independent runs.

## Crossing and uncertainty

The nine seed-mean contrasts receive an equal-weight non-increasing isotonic
fit. The point estimate is the first zero crossing, linearly interpolated only
inside its adjacent tested grid interval. It is an estimated crossing, not an
exact or causal threshold.

Every raw adjacent higher-minus-lower seed contrast is also reported. An
upward step is a material reversal only when its paired-seed two-sided
Student-t 95% interval lies wholly above zero. A unique primary bracket is
reported only when there is no material reversal and the raw mean curve has
exactly one positive-to-nonpositive adjacent bracket. Otherwise the primary
result is the list and union of raw candidate brackets, and the scalar isotonic
root is labelled as sensitivity analysis.

Whole seed profiles are resampled, preserving dependence across all strengths.
Each bootstrap profile is refit isotonicly. Draws that cross before the grid or
not by its end are retained as left- or right-censored. The reported percentile
interval uses the extended-real ordering, so material censoring cannot be
silently discarded. The 95th percentile of the bootstrap maximum absolute
deviation gives a simultaneous band across all nine raw mean contrasts.

The analysis JSON includes the raw seed contrasts, pointwise intervals,
simultaneous bands, isotonic estimates, changed-label summaries, crossing
bracket, censoring probabilities, and plot-ready rows.

## Command after both campaign seals validate

```bash
python -B -m experiments.gradient_reuse_crossover_analysis.analysis \
  --campaign pmnist \
  --historical-aggregate path/to/corrected/pmnist/AGGREGATE.json \
  --followup-aggregate path/to/crossover/pmnist/AGGREGATE.json \
  --output path/to/crossover/pmnist/ANALYSIS.json

python -B -m experiments.gradient_reuse_crossover_analysis.analysis \
  --campaign cifar \
  --historical-aggregate path/to/assignment_strength/aggregate.json \
  --followup-aggregate path/to/crossover/cifar/AGGREGATE.json \
  --output path/to/crossover/cifar/ANALYSIS.json
```

The default bootstrap uses 20,000 draws and a fixed analysis seed. Each output
binds the two input aggregates by SHA-256 and refuses to overwrite an existing
analysis file.
