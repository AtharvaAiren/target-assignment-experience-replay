# Prespecified scalar-study protocol

This study follows the completed, source-frozen preliminary audit. It does not
tune or confirm a new adaptation method. It resolves two questions left by the
preliminary result:

1. Is the Beijing previous-gradient benefit target-specific, or can generic
   shrinkage, output-bias persistence, or shuffled past labels explain it?
2. Is the functional-OT implementation capable of helping when heterogeneous
   mobility utility really exists, even though the natural-stream oracle found
   essentially no allocation opportunity beyond uniform mobility?

## Online timing and causality

For packet `t`, rows K0--K6 are snapshotted before reading any target from
packet `t`.  They may use model state and labels revealed through packet
`t-1`.  K7--K9 are explicitly nondeployable cross-fitted diagnostics.  Every
branch is virtual and discarded.  The common center receives exactly one
ordinary optimizer update per packet after all predictions are fixed.

All directions use the same radius: the median L2 norm of the previous 16
actual center updates.  No row receives extra training updates, replay, raw
storage, or optimizer exposure.

## Target-specificity rows

* K0: zero virtual displacement.
* K1: previous revealed full gradient.
* K2: radial shrinkage toward the parameter origin.
* K3: the output-bias component of the previous gradient only.
* K4: a previous-packet gradient computed after deterministically shuffling
  that already-revealed packet's labels.
* K5: K1 after removing output-bias coordinates and its radial component.
* K6: a deterministic matched-norm isotropic direction independent of labels.
* K7: current-packet event-disjoint two-way cross-fitted oracle.
* K8: K7 with targets shuffled independently inside each support fold.
* K9: reverse K7.

The primary target-specific verdict is locked to Beijing.  It passes only if
K1 beats every trivial control by at least 1% relative NRMSE with paired CI
above zero, K5 beats K0 by at least 1% with CI above zero, and at least three
of four development seeds win.  Metro is a locked negative-control stream and
must be noninferior within 1%.  Appliances Energy and Tetouan Power Consumption
are breadth streams, not replacements for the primary dataset.

Development and confirmation summaries are both reported.  Confirmation runs
regardless of development sign, so a negative result is independently
measured rather than inferred from a failed development gate.

## Corrected synthetic control

The old synthetic generator controlled latent-optimum velocity but did not
control the observed useful gradient.  The replacement directly constructs
normalized observed gradients with correlations `+0.9`, `0`, and `-0.9`.
It must empirically realize those correlations and make previous, zero, and
reverse directions optimal in the appropriate regimes.  This validates the
forecast audit; it is not evidence about any natural stream.

## Natural breadth and robustness

Every natural stream remains in timestamp order.  Scaling statistics use only
the leading 5% calibration prefix.  Scoring clusters are calendar months for
the original Metro/Beijing loaders and calendar weeks for the shorter energy
streams.  The locked primary learner is a linear base plus 16 ReLU functional
units optimized by AdamW with packet size 32.  Characterization additionally
uses a linear model, SGD with momentum, and packet sizes 16 and 64 on Beijing
and Tetouan.  These rows cannot redefine the primary result.

## Functional-OT positive control

An eight-atom controlled environment has an interior mobility target that
moves locally along a known functional line geometry.  At each step, demand is
analytically constructed so that this target is a first-order optimum of the
locked proximal objective under the correct geometry.  The identical demand,
previous allocation, box/simplex constraints, and numerical budget are then
given to the correct functional cost and to a spectrum-matched shuffled cost;
uniform allocation is the third control.  This is deliberately an engineered
solver sanity check, not a causal natural-stream experiment.  It passes only
if functional cost beats uniform by 20% and shuffled cost by 5%, with a paired
seed CI above zero and at least six of eight seed wins.  Failure means the
implementation cannot support even the narrow positive-sanity claim; success
does not override the natural-stream no-opportunity result.

## Inference and reporting

Natural effects use paired two-way inference over initialization seeds and
circular moving blocks of ordered time clusters.  Synthetic controls use
paired IID seed bootstrap.  We report absolute and relative NRMSE, confidence
intervals, temporal-cluster and seed wins, branch resource counts, data hashes,
and all locked rows regardless of outcome.
