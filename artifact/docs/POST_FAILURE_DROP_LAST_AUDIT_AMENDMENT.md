# Post-failure `drop_last` audit-field amendment

- Recorded: `2026-10-13`
- Trigger: the first PMNIST replay pilot completed training, then failed during
  post-run assignment-audit validation.
- Scope: one omitted effective-argument field in the audit producer.
- Scientific locks, datasets, task/order banks, seeds, arms, hyperparameters,
  estimands, inference families, and execution widths: unchanged.

## Observed failure

The original v1 source ran the locked Mammoth command with explicit
`--drop_last 0`.  The bootstrap independently validated Mammoth's parsed
runtime value as `False`, so the partial-final-minibatch training behavior was
correct.  After training, the assignment-strength audit producer omitted
`drop_last` from its effective-argument record.  The fail-closed consumer then
reported the missing value as `None`:

```text
RuntimeError: assignment effective argument changed: drop_last=None, expected=0
```

No partial scientific metric was inspected in diagnosing this failure.

## Bounded correction

The amended producer adds `drop_last` to its existing effective-argument
allowlist.  It does not alter training, parsing, batching, optimization, or
aggregation.  A regression test now constructs the real intervention audit
and requires the emitted effective-argument record to contain
`drop_last=False`.  The consumer remains strict; it is not weakened.

The original v1 handoff, installed source, and result roots are retained as
failure evidence and are excluded from all aggregation.  No v1 unit seal is
reused.  Both campaigns restart from unit 1 in fresh v2 source, output, and
coordinator namespaces, so all 816 units share one amended source identity.

## Retained v1 identities

- Remote source: `gradient_tmlr_exact_reference_source_20261012_v1`
- PMNIST output: `results_gradient_tmlr_exact_reference_pmnist_20261012_v1`
- Source-bundle SHA-256:
  `d33c9b40e9985b31d4c0d809da84042589118c3114c165861104fd86e6141131`
- External source-manifest SHA-256:
  `25825658fde9ce26b2adf05cddf2dbc9550cef3d4cb84ae8634f3e21a0e686b7`
- Source-archive SHA-256:
  `c541b1bfb9bcd8692411e678e571f0db4ef8a36ece05747bebec476ce81b7926`

The v2 identities are recorded by its authenticated source manifest and
handoff checksum ledger; they are intentionally not embedded here because
this amendment is itself part of the source closure.
