# Provenance and public-source policy

The repository is a fresh anonymous review release assembled from campaign
exports whose retained records contain execution-time hashes and completion
status. The omitted original run trees cannot be re-authenticated from this
compact release. This is not a copy of a development Git history.

The release process was restricted to replacing machine-local absolute paths,
account names, and private host strings in serialized metadata with portable
`artifact://` references. For every changed JSON/JSONL file,
`provenance/SANITIZATION_MANIFEST.json` records the author-retained original
SHA-256, public SHA-256, and replacement count. The original bytes are not
distributed, so this bridge cannot be independently replayed from the public
tree alone. A sanitized public file is not byte-identical to its
execution-time counterpart; the manifest records the claim that numerical
scientific fields were unchanged.

Operational recovery scripts, launch logs, queue states, failures, and PID
files are excluded. Where a campaign required a non-scientific portability or
numerical correction, the terminal report retained here records whether any
scientific outcome was modified. `SHA256SUMS` authenticates its listed public
files; the release-archive builder uses that list as the complete archive
allowlist.

Most manifests in `provenance/source/` bind the source identity stated in their
own records. The two original mechanism manifests are deliberately retained as
immutable execution-time records. Their anonymous public counterparts rename
one inert linked-project isolation sentinel in two locks and its fail-closed
validator; they therefore are not presented as byte-identical execution
source. `provenance/source/PUBLIC_SANITIZATION.json` maps the three historical
digests to their public digests and separately identifies the verification
code updated to understand this boundary. `scripts/verify_results.py` checks
that every unlisted historical source byte is unchanged, the public digests
match, and completed results still cite the historical lock hashes. The
private identifier required to reconstruct the original three bytestrings is
intentionally not distributed.

Other release sanitization is recorded in
`provenance/SANITIZATION_MANIFEST.json`. These author-recorded bridges do not
turn sanitized public files into execution-time bytes, and they do not rewrite
any frozen result identity.

The original continual-primary aggregate authenticates 288 execution units by
unit, completion, and artifact-bundle hashes.  Those full execution directories
were not included in the review release.  To make the numerical reduction
auditable without inventing missing evidence,
`results/continual_primary/unit_endpoints.jsonl` is a deterministic, lossless
projection of that frozen aggregate. It retains the three endpoint values and
applicable diagnostic scalars represented there for every unit and copies the
three execution hashes recorded in the frozen run manifest. The companion
manifest binds the projection to the byte-identical aggregate, and the verifier
reproduces the complete aggregate exactly. Because the table was projected from
the aggregate rather than recovered from the missing unit directories, placing
an endpoint beside a recorded unit hash does not independently authenticate
that endpoint against the omitted bytes. The verifier cannot recompute those
hashes or inspect omitted training traces.

The legacy Permuted-MNIST reference arms have a separate scientific boundary.
The pinned dataset and replay buffer both consumed process-global NumPy state,
so independently launched replay and no-replay arms were not guaranteed to see
the same later task permutations. The public release preserves those values but
treats Permuted-MNIST reference-arm contrasts and direct cross-method rankings
as historical descriptive outputs. Matched paired-versus-reassigned replay arms
retain the relevant replay RNG-consumption schedule. A separately locked fresh
campaign uses RNG-isolated, content-hashed domain banks and is retained under
`results/corrected_reference_order/pmnist`; it resolves the reference question
for that campaign without changing the status of the historical arms. The
independent CIFAR record in the same result family covers three task-block and
three fully regrouped class orders.

The compact release retains each fresh campaign's aggregate, binding, and
terminal completion seal rather than its raw unit tree. The PMNIST campaign is
bound to the post-outcome audit-adapter seed amendment, and the fresh CIFAR
restart is bound to the later multiline-log parser amendment. Both amendments
are disclosed in `docs/`. Neither changed a scientific lock; the affected
queues restarted from unit 1 and no incomplete unit entered an aggregate.
The current executable corrected-reference source descends from the final
CIFAR source snapshot but is not globally byte-identical to it. Two
manifest-bound documentation files were rewritten after completion to describe
the observed campaign state, and the shared intervention later acquired the
exact-reference `drop_last` audit field. The PMNIST manifest additionally
predates the CIFAR-only parser amendment. These bounded per-campaign
differences are enumerated and checked by `scripts/verify_results.py`.

The crossover source has its own historical boundary. Its manifest binds 42
execution-time files, three of which differ from the later shared tree. The
portable `provenance/crossover/execution-source.patch` transforms a disposable
copy of the current tree into those exact historical bytes. The verifier
requires exactly the documented three differences, applies the patch, and
checks every manifest digest. The historical manifest itself is not rewritten.

The later exact-reference follow-up is retained under
`results/exact_reference/`. Its 240-unit PMNIST and 576-unit CIFAR aggregates,
bindings, terminal seals, and combined seal are byte-identical to the files in
the authenticated server export. The raw unit tree and operational logs are
not distributed in this compact release. The execution-time source manifest
is retained at `provenance/exact_reference/source-manifest.json`; the public
verifier checks its own hash, its canonical source-bundle hash, both locked
configuration hashes, and every listed source file. An initial pilot stopped
before aggregation because the audit serializer omitted the already applied
`drop_last=0` argument. The audit-only amendment and regression test are part
of the retained source closure, and the final campaigns restarted from unit 1
under fresh output identities.
