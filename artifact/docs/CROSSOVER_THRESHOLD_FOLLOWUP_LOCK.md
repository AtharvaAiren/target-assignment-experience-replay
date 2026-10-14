# Dense reassignment crossover follow-up

Status: outcome-informed design, prospectively locked before reading any result
from the dense follow-up.

## Why this study exists

The completed dose-response study tested requested strengths 0, 0.25, 0.50,
0.75, and 1.00. The 0.75 point was important because it was specified before
that study ran, not because 75% is a scientifically privileged threshold. The
completed outcomes place the apparent equality with conventional no replay
between the last two points. This separate study localizes that crossing.

The design is explicitly outcome-informed. It does not replace, enlarge, or
retroactively redefine the original confirmatory families.

## Discrete intervention grid

With a replay minibatch of 32, the intervention uses

`m = 1 + round_half_up(lambda * 31)`

selected rows. The old 0.75 and 1.00 requests select 24 and 32 rows. There are
only seven genuinely new supports between them. They are locked as follows:

| Requested bp | Selected rows | Selected fraction | Induced covariance-erasure strength |
|---:|---:|---:|---:|
| 7,742 | 25 | 25/32 | 24/31 |
| 8,065 | 26 | 26/32 | 25/31 |
| 8,387 | 27 | 27/32 | 26/31 |
| 8,710 | 28 | 28/32 | 27/31 |
| 9,032 | 29 | 29/32 | 28/31 |
| 9,355 | 30 | 30/32 | 29/31 |
| 9,677 | 31 | 31/32 | 30/31 |

The induced strength, not the rounded request, is the primary horizontal axis.
The observed fraction of labels whose values actually change is reported as a
separate diagnostic because fixed points and equal-label collisions make it
smaller than the selected-row fraction.

## New runs

- Permuted-MNIST: seeds 3801--3812, the same RNG-isolated fixed task bank as
  the completed corrected-reference study.
- Split CIFAR-100: seeds 3501--3512, the same default class stream and task-IL
  primary endpoint as the completed assignment-strength study.
- Seven new strengths, four nested assignment replicas, and 12 run seeds give
  336 new units per benchmark and 672 total.
- Conventional no-replay controls and the 0.75/1.00 endpoints are not rerun.
  They are reused only from hash-bound completed aggregates.
- All 336 units in a benchmark are required before its scientific aggregate is
  generated. Monitoring exposes only completion counts and process state.

## Estimand and uncertainty

For each seed and strength, the four assignment replicas are averaged first.
The primary response is reassigned-ER final average accuracy minus the
same-seed conventional-no-replay accuracy. The primary estimate is the first
tested discrete support at which the expected contrast is nonpositive, stated
as the adjacent tested support bracket. A nonincreasing isotonic curve is shown
alongside every raw grid mean. A within-bracket linear zero is secondary and is
described as model-based, never as an exact percentage.

Whole seed profiles are resampled with common seed indices across all nine
points. This preserves across-strength dependence and keeps the four assignment
replicas nested within seed. Bootstrap draws with no crossing by full
reassignment remain right-censored above 1.00 rather than being discarded.
The analysis also reports a simultaneous band across the complete grid.

An adjacent increase is called material only when its across-seed paired,
two-sided Student-t 95% interval lies entirely above zero. A unique primary
bracket is issued only if there is no such reversal and the raw means have
exactly one positive-to-nonpositive adjacent bracket. Otherwise the primary
result is the list or union of candidate raw sign-change brackets; an isotonic
scalar root is then sensitivity analysis only. Permuted-MNIST is analyzed
primarily against conventional no replay and separately against the
fixed-current reference.

## Stopping and amendments

There is no outcome-based early stopping. Any correction made after a new
follow-up outcome becomes readable requires a new versioned sensitivity run;
completed PASS units remain immutable.
