# Post-outcome multiline Mammoth-log parser amendment

- Recorded: `2026-10-03`
- Trigger: the first Split CIFAR-100 unit completed training, after which
  post-run metric extraction failed before the unit was sealed
- Scope: parsing of multiline Mammoth `logs.pyd` records
- Scientific locks, datasets, class orders, seeds, arms, hyperparameters,
  estimands, and inference families: unchanged

## Observed failure

The adapter-seed amendment campaign completed and sealed all `228`
Permuted-MNIST units. The coordinator then started the Split CIFAR-100 queue.
The first CIFAR unit completed its Mammoth execution, but
`parse_mammoth_log` selected only the final physical line of the result
record. CIFAR's logged NumPy arrays were sufficiently large for their
representations to span multiple lines. The selected suffix therefore began
inside an array and raised:

```text
SyntaxError: unmatched ']'
```

The failure occurred during metric extraction after training. The unit was
not sealed, the CIFAR aggregate was not constructed, and no partial CIFAR
record is admitted to the final campaign.

## Bounded correction

The parser now parses the complete log as a Python module, requires every
top-level statement to be an expression, and evaluates only the final
expression through the existing restrictive AST evaluator. This supports
both a single multiline record and several appended expression records. It
does not execute the log and does not broaden the accepted call grammar.

Two focused regression cases were added:

1. a multiline NumPy-array record following an older record is parsed as the
   final record;
2. any non-expression statement is rejected.

The retained failed CIFAR logs must pass a read-only parser and 20-task metric
reconstruction check before a new queue is launched. Those files are used
only as a compatibility test and are never promoted into a sealed unit.

## Continuation boundary

The authenticated `228/228` Permuted-MNIST campaign remains unchanged under
the adapter-seed source bundle. Repeating it would add no protection against
this CIFAR-specific serialization case. The incomplete CIFAR root is retained
unchanged and excluded. All `504` CIFAR units restart from unit 1 in a fresh
result root under the multiline-parser source bundle. Final validation keeps
the two campaign-level source identities explicit rather than pretending that
one source bundle produced both campaigns.

## Source identities

Completed Permuted-MNIST campaign:

- Source-manifest SHA-256:
  `b051a0f2f08152eda2cb0d790871d8b7eeb5ab5bcad2f6de305c13590d4e1079`
- Source-bundle SHA-256:
  `ace58c1fa189e8d13ed8e4a2da54103add8f9407ebcaee3965369ae5cb5c465f`

CIFAR restart:

- Clean 27-file source-closure SHA-256:
  `4b42b1165ea2605d6f14937f49a9bed04e6ae1061f71b1101bd4826f81ead676`
- Source-manifest SHA-256:
  `9baf8c13506f9a53210e5ff79a78fba130013e9fe91e5bd505b246ec17552ca1`
- Source-bundle SHA-256:
  `2d03778d876fa53481756a739c8dda0ed7fa1c535138540d9b077e94a0f65bbd`

The new archive contains exactly the 27 manifest-bound files and no
AppleDouble companions or symbolic links.
