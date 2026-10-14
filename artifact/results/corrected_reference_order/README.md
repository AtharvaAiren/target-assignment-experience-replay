# Corrected-reference and order-robustness results

This directory contains the compact public records from two independent,
prospectively locked follow-up campaigns:

- `pmnist/`: 228/228 completed units over seeds 3801--3812, with an
  RNG-isolated Permuted-MNIST domain bank, two no-replay controls, paired ER,
  and four nested assignment replicas at each nonzero reassignment dose.
- `cifar/`: 504/504 completed units over the same 12 seeds, three locked
  task-block orders, three locked fully regrouped class orders, two no-replay
  controls, paired ER, and four nested fully reassigned replicas.

Each campaign directory retains its execution-time `CAMPAIGN.json` binding,
`AGGREGATE.json`, and terminal `DONE.json`. The digest of each aggregate is
bound by its completion record. The public package omits the raw unit trees,
so it can check these bindings and manuscript-facing aggregate invariants but
cannot reconstruct every unit locally. The author-retained complete trees were
validated unit by unit before these compact records were exported.

The PMNIST and CIFAR campaigns intentionally have different source-manifest
identities. The PMNIST queue completed after the bounded audit-adapter seed
amendment. A later multiline-log parser amendment affected only the CIFAR
restart. Both disclosures are retained in `docs/`, and their source manifests
are under `provenance/corrected_reference_order/`.

The executable files in the current public tree match the final CIFAR
execution manifest. Its two documentation entries were rewritten after
completion to replace prospective wording with the completed status. The
PMNIST manifest records the earlier adapter-seed-amended source, before the
CIFAR-only parser repair. A fresh reproduction must generate its own source
manifest from the current tree.
