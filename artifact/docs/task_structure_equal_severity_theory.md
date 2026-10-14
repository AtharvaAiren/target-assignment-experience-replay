# Equal-severity task-structure study: estimand and covariance guard

## Intervention

For one sampled replay batch, let row `i` have input `x_i`, global class
target `y_i`, and historical task `h_i`.  Under pinned Mammoth commit
`e75a491c69fd729edeb01431afb753d9157d9a81` with `seq-cifar100` and the
`20tasks` configuration, `h_i = y_i // 5`.  Class permutation, custom class or
task order, and label noise are forbidden and authenticated at runtime.

Selection uses no prediction, loss, gradient, validation metric, or scientific
outcome.  A counter-keyed RNG supplies tie priorities.  The selector first
forms the maximum number of disjoint unequal-class row pairs within each
historical task, then greedily matches pair pools from different tasks to the
maximum feasible number of disjoint quadruples.  Each selected quadruple is

\[
Q=((i_1,i_2),(j_1,j_2)),\qquad
h_{i_1}=h_{i_2}\ne h_{j_1}=h_{j_2},
\quad y_{i_1}\ne y_{i_2},\quad y_{j_1}\ne y_{j_2}.
\]

Split CIFAR-100 tasks have disjoint global class sets, so all four labels are
distinct across the two task pairs.  The within-task arm swaps `i_1 <-> i_2`
and `j_1 <-> j_2`.  The across-task arm swaps the two pair blocks, with a
counter-keyed orientation.  Therefore both arms:

- operate on exactly the same selected inputs and original targets;
- change exactly four target values per selected quadruple;
- preserve the exact replay-batch target multiset;
- leave every unselected replay row unchanged.

The primary path estimand is final Task-IL accuracy in the across-task arm
minus final Task-IL accuracy in the within-task arm, paired by 12 fresh run
seeds.  Class-IL is mandatory.  An unchanged ER probe uses true replay targets
for training and computes all three gradients from one common state.

## The grouped covariance identity and its scope

The following identity explains what a *uniform random group permutation*
would average over.  It is not an identity for this study's forced swaps.

At a fixed parameter state, for scalar squared loss write row Jacobians as
`J_i`, predictions as `f_i`, and

\[
g(y)=\frac1n\sum_i J_i^\top(f_i-y_i).
\]

Let `G` be a fixed partition and let `pi_G` be uniform over every permutation
that keeps targets inside each group.  With `bar y_g` the target mean in group
`g`,

\[
\mathbb E_{\pi_G}g(y_{\pi_G})
=\frac1n\sum_i J_i^\top(f_i-\bar y_{g(i)}),
\]

and hence

\[
g(y)-\mathbb E_{\pi_G}g(y_{\pi_G})
=-\sum_g\frac{n_g}{n}\operatorname{Cov}_g(J,y).
\]

For softmax cross-entropy the target-dependent part of the raw parameter
gradient is also linear after replacing scalar `y_i` by its one-hot vector and
`J_i` by the logit Jacobian.  Thus an analogous fixed-state expectation
identity exists for a genuinely uniform group permutation law.

It would be incorrect to substitute either expectation for the intervention
implemented here.  The present selector conditions on unequal classes,
selects disjoint pairs by a greedy maximum-coverage rule, forces every chosen
row to change target value, and matches two task pools.  Neither the
within-task nor across-task mapping is uniform over all group-preserving or
global permutations.  The full-training arms are additionally path-level and
cease to share a parameter state after the first eligible update.

Accordingly, the probe reports the exact realized, unnormalized full-replay
cross-entropy gradients and their norms, distances, dot products, and cosines
as descriptive common-state diagnostics.  It makes no grouped-covariance
expectation claim and no causal decomposition claim.
