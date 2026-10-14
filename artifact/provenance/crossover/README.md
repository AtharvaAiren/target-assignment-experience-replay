# Crossover provenance

`PREOUTCOME_SEAL.md` records the v2 amendment and confirms that v1 stopped in
strict preflight before any outcome-producing process or output root existed.
The only recorded environment-manifest change was the Linux kernel string and
its derived environment digest; the design, seeds, support grid, analysis, and
stopping rule did not change.

`crossover-source-manifest.json` records the exact sealed 42-file scientific
source closure. The shared public tree later acquired three bounded changes:
the dense support grid was removed from two shared assignment-strength files,
and the exact-reference audit added a `drop_last` regression hook and test.
Consequently, the retained historical manifest does not validate the shared
tree without reconstruction.

`execution-source.patch` transforms a fresh copy of the current public tree
into the exact manifest-bound crossover source for those three files. Apply it
only in a disposable copy with `patch -p1`; do not modify this release tree or
rewrite the historical hashes. `scripts/verify_results.py` checks that the
current tree differs from the historical manifest at exactly those three paths
and that applying the patch reconstructs all 42 recorded digests.

The author-side tar container and handoff checksum sheet are excluded from the
anonymous public release because their container metadata contains
machine-local owner strings. The portable reconstruction patch, manifest,
locked configuration, and public release inventory provide the review-stage
source boundary without distributing that container.
