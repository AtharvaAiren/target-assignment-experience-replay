# Post-outcome audit-adapter seed amendment

- Recorded: `2026-10-02`
- Trigger: the first replay unit completed all 20 Permuted-MNIST tasks, then
  failed during post-run audit summarization
- Scope: one missing metadata field in an internal adapter
- Scientific locks, datasets, task banks, order banks, seeds, arms,
  hyperparameters, estimands, and inference families: unchanged

## Observed failure

The original source closure successfully sealed the first two no-replay units.
The third unit, the seed-3801 paired-replay arm, completed training and wrote
its Mammoth metrics, assignment audit, and bootstrap audit. Its post-run
integrity check then raised:

```text
AttributeError: 'types.SimpleNamespace' object has no attribute 'seed'
```

The corrected-campaign runner adapts its `AuditUnit` or `CifarUnit` to the
shared assignment-audit summarizer. The adapter supplied `method`,
`lambda_bp`, `benchmark`, and `mammoth_dataset`, while the summarizer also
requires `seed` to authenticate every audit row's `run_seed`. The training
path does not read this adapter; the omission affects only post-run validation
and sealing.

## Bounded correction

The implementation change adds exactly this field to the adapter:

```python
seed=unit.seed
```

A focused regression test now requires the adapter to carry the method,
strength, benchmark, dataset, and locked run seed into the shared summarizer.
The amended source manifest explicitly identifies itself as a post-outcome
implementation amendment rather than a pre-outcome source lock.

The original aborted result root is retained unchanged and excluded from the
scientific aggregate. Because its first two units bind the original source
bundle, none of them is reused. Both campaigns restart from unit 1 in fresh
output roots under the amended source bundle. This prevents mixed-provenance
aggregation.

## Source identities

Original identities retained for provenance:

- Source-closure SHA-256:
  `c565a6a43d28d18665667357ccd43c17087487d950520b5ea7e58e4ae0aa45dc`
- Source-manifest SHA-256:
  `c49e7d40f166b57600407b47d97be4fcbea9de856691dd7b8d44e9229b96c4d0`
- Source-bundle SHA-256:
  `9ea3688382200da29b02abde8f0ba2b39d9452a84a4b5f72523d89b8d1a779a3`

Amended identities:

- Clean 27-file source-closure SHA-256:
  `9beac56b37ec0b7e544b36ba839e1cffdf9f1eef9a1e96b1028af119ebcae98b`
- Source-manifest SHA-256:
  `b051a0f2f08152eda2cb0d790871d8b7eeb5ab5bcad2f6de305c13590d4e1079`
- Source-bundle SHA-256:
  `ace58c1fa189e8d13ed8e4a2da54103add8f9407ebcaee3965369ae5cb5c465f`

The amended archive contains exactly the 27 manifest-bound files and no
AppleDouble companions. Its contents independently match the amended manifest
and source-bundle digest. Server-side focused tests and a read-only validation
of the retained replay audit are required before restarting the full campaign.
