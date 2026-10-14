# Claim-to-artifact map

The manuscript's central quantities are total reuse value, value retained
after reassignment, and correct-pairing value. The table below identifies the
authoritative public evidence for each reported family.

| Manuscript claim or analysis | Authoritative artifact |
|---|---|
| Shallow four-stream primary results and controls | `results/scalar/*.json` |
| Repeated-permutation and local diagnostics | `results/reviewer/CAMPAIGN_SUMMARY.json`, `results/reviewer/summaries/*.json` |
| CNN/GRU breadth | `results/breadth/PAPER_SUMMARY.json` and CSV companions |
| Main continual-learning panel | `results/mechanisms/SUMMARY.json`; `results/continual_primary/unit_endpoints.jsonl` is a lossless 288-row endpoint projection of that aggregate and cross-references the unit hashes recorded in `run_manifest.json` |
| Five-seed pooled inference and held-out replication | `results/primary_seed_completion/FIVE_SEED_PRIMARY_SUMMARY.json`, `HELD_OUT_REPLICATION.json` |
| Reviewer extension | `results/reviewer_extension/units.jsonl`, `secondary_endpoints.json` |
| Original Pythia stress-test window | `results/transformer/RESULT.json` |
| Amended Pythia stress-test window | `results/reviewer_extension/pythia_stream.json` and `pythia_completion.json` |
| Partial-reassignment dose response | `results/assignment_strength/aggregate.json` |
| Dense crossover localization and censored bootstrap analysis | `results/crossover/{pmnist,cifar}/ANALYSIS.json`; compact aggregates, campaign bindings, and completion records are colocated |
| Conditional-mean comparison | `results/mean_matched/aggregate.json` |
| Within/across-task equal-exposure comparison | `results/task_structure/aggregate.json` |
| Task-relative localization | `results/task_relative/aggregate.json` |
| Shared-output and variance-ladder controls | `results/explanatory/aggregate.json` |
| Corrected PMNIST reference and CIFAR order robustness | `results/corrected_reference_order/{pmnist,cifar}/AGGREGATE.json`; campaign bindings and completion seals are colocated |
| Exact stepwise coefficient-matched reference | `results/exact_reference/{pmnist,cifar}/AGGREGATE.json`; campaign bindings, completion seals, the combined 816-unit seal, and `provenance/exact_reference/source-manifest.json` provide the compact evidence boundary |
| Exact finite-horizon analysis | `results/finite_horizon/result.json` and 60 unit records |
| Appendix controllers | `results/controller/*_summary.json` |

`scripts/verify_paper_values.py` checks the study counts and headline
invariants directly against these files.  As part of that command,
`scripts/verify_continual_primary_reconstruction.py` verifies all 288 compact
rows, reconstructs the 54 seed rows and 18 method summaries exactly, and checks
the 18 three-decimal rows reported in the manuscript.  This is an endpoint
reconstruction, not a replacement for the omitted execution-time unit
directories and training traces; that residual boundary is recorded in
`results/continual_primary/unit_endpoints_manifest.json`.

## Legacy Permuted-MNIST reference limitation

The pinned Permuted-MNIST implementation generates task permutations from the
process-global NumPy RNG, which is also consumed by replay-buffer operations.
Separately launched replay and no-replay arms therefore were not guaranteed to
receive identical later task permutations even when their nominal seed agreed.
For the frozen legacy panel, Permuted-MNIST `V_total`, `V_perm`, buffer-size
reference contrasts, and direct cross-method rankings against a reference arm
are retained only as historical descriptive outputs. Within a replay method,
the matched paired and target-reassigned arms preserve the relevant replay RNG
consumption schedule, so their `V_pair` comparison and the replay-only
dose/structure/common-state analyses do not inherit this particular
reference-arm defect. The compact endpoint table
preserves the frozen record; it does not repair or re-authenticate that design.
The separately locked fresh PMNIST aggregate uses RNG-isolated domain banks
and supports reference-arm contrasts for its own ER campaign. It does not
alter the legacy limitation. The corresponding CIFAR aggregate reports the
separate six-order robustness audit. Their compact public records omit the raw
unit trees, so public verification checks aggregate identities, terminal
bindings, counts, and manuscript-facing values rather than reconstructing all
732 corrected-reference/order units. The dense crossover release likewise
omits its 672 raw unit directories while retaining the two 336-unit aggregates,
bindings, completion records, deterministic analyses, and source manifest.
The exact-reference release follows the same compact boundary: it retains the
two aggregates, campaign bindings, completion seals, combined 816-unit seal,
and execution-time source manifest, but omits the 816 raw unit directories.
The public verifier therefore checks hashes, identities, counts, source
closure, and manuscript-facing values rather than reconstructing either
aggregate from raw units.

The crossover execution source is retained as a historical manifest plus a
portable reconstruction patch because three shared files changed later. The
verifier requires exactly those three differences and validates the
reconstructed 42-file closure; this does not turn the omitted raw unit tree
into public evidence.
