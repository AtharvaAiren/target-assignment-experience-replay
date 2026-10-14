# Exact stepwise coefficient-matched reference

This directory contains the compact public evidence for the prospectively
locked fresh-seed follow-up. The current-only control uses the same scalar
coefficient on the current minibatch loss that replay would use at every
optimizer step, including buffer warm-up and retained partial minibatches. It
does not sample replay examples or advance replay-sampling random-number state.

The two campaigns use seeds 4101--4112:

- `pmnist/`: 240 completed units, with four assignment replicas averaged
  within seed before inference;
- `cifar/`: 576 completed units, with assignment replicas and three orders per
  family averaged within seed before inference.

Each campaign directory retains the byte-identical aggregate, locked campaign
binding, and terminal completion seal from the authenticated export.
`COMBINED_DONE.json` binds both campaign seals and records 816 completed units.
The execution-time source manifest is retained at
`provenance/exact_reference/source-manifest.json`.

The compact review release intentionally omits the raw unit directories and
training logs. Public verification therefore checks the aggregate hashes,
campaign bindings, completion counts, source closure, and manuscript-facing
values; it does not reconstruct the aggregates from raw unit records. The
author-retained authenticated export contained 8,612 payload files plus its
manifest and had outer SHA-256
`4523569cd0c8b3619a6cda05c767638a156d4bae975769005e981026f6d3af0a`.

Headline results are reported in percentage points with the training seed as
the inferential unit. Paired replay minus exact current-only is 29.97 on
PMNIST and 27.29/27.27 on task-aware CIFAR for the task-block/full-class order
families. Fully reassigned replay minus exact current-only is -23.75 on PMNIST
and -10.77/-10.49 on task-aware CIFAR. These results remove the scalar
current-loss-weight mismatch, but they do not match replay gradients, batch
composition, batch-normalization state, or the subsequent optimization path;
they are not a mechanistic decomposition.
