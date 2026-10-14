# Portable Transformer compatibility closure

The original Pythia campaign required three compatibility corrections after
fail-closed preflight checks. The execution-time patch files embedded absolute
server paths, so those bytes are not part of the anonymous artifact.

This release supplies a path-free author reimplementation of the three
documented rules:

- `experiments/gradient_reuse_reviewer_extension/transformer_correction.py`
- `provenance/transformer_correction_portable.json`
- `provenance/transformer_correction_history.json`

The history bridge preserves author-recorded SHA-256 identifiers for the
historical manifests and executable patches. The historical bytes are absent,
so those identifiers cannot be independently checked from this release. The
bridge explicitly records that the public source is not byte-identical to the
historical patches. The portable manifest authenticates only the public
implementation and bridge. It does not prove semantic equivalence to the
historical patches.

The portable implementation is intended to enforce these three documented
rules:

1. The unused `classifier_dropout=0.1` field is tolerated only for the
   authenticated GPT-NeoX causal-LM architecture, after rejecting active
   dropout modules, BatchNorm, nonzero causal-LM dropout fields, and classifier
   modules.
2. The base tokenizer vocabulary, complete tokenizer length, and model
   input/output rows must be exactly 50,254, 50,277, and 50,304. The complete
   50,304-way model softmax is preserved.
3. Token IDs are checked against the exact 50,277-ID effective tokenizer
   inventory, including the 23 authenticated added tokens. IDs belonging only
   to padded model rows are rejected. No token is remapped and no record is
   skipped.

`transformer_bootstrap.py` uses this bundled implementation by default. When
all other external E7 inputs are supplied, a fresh run binds the portable
manifest and implementation bytes in its live input bundle. Fresh portable
executions use correction-provenance schema v2, which records the executed
public digests separately from the author-recorded historical digests and sets
`semantic_equivalence_verified` to false. The frozen historical result retains
its legacy v1 provenance record unchanged. This repository does not distribute
the complete parent campaign and therefore does not by itself provide a full
live E7 execution closure.

The focused CPU tests are:

```bash
python -m pytest tests/test_transformer_portable_correction.py
```

These tests exercise the intended narrow classifier-dropout exception,
padded-vocabulary dimensions, effective added-token IDs, padded-row rejection,
portable-byte authentication, and anonymity of all three public files. They
are tests of the documented portable behavior, not evidence that it is
semantically equivalent to the unavailable historical patch bytes. They also
do not replace the external checkpoint, corpus, parent campaign, and runtime
needed for a full Pythia run.
