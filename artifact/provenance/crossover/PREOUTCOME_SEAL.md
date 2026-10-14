# Pre-outcome seal: dense crossover localization follow-up, v2 amendment

Seal date: 2026-10-08 (Asia/Kolkata)

Status: `OUTCOME_INFORMED_PROSPECTIVELY_LOCKED_BEFORE_CROSSOVER_FOLLOWUP_OUTCOMES`

This v2 seal supersedes the immutable v1 crossover handoff before scientific
execution. The v1 pre-outcome seal SHA-256 is
`c148cec2861cfd21494cee5b221ef64a5eb179597b5eda6031b7cccbe21dbfd5`.
The v1 server attempt stopped during strict environment validation because the
server kernel had changed from 6.8.0-142 to 6.8.0-146. Validation failed before
an output root was created and before any crossover unit subprocess or other
outcome-producing process started. Consequently, v1 started and completed zero
crossover units; no unit completion seal, campaign aggregate, or campaign DONE
seal existed. No crossover outcome---including partial metrics, scientific
stdout, checkpoints, or failed-attempt results---was inspected or used. Only
operational preflight diagnostics were examined.

The v1 local package and failed-attempt evidence remain unchanged and are
classified as `SUPERSEDED_BEFORE_EXECUTION`. V2 uses new handoff, source,
result, phase, PID, log, archive, and fetched-result paths. No v1 unit may be
reused, copied, or combined with v2. The global GPU lock is intentionally
shared across versions to prevent concurrent execution.

## Environment amendment

The sole difference in the recorded environment manifest is
`observed.platform`; the environment-bundle digest changes as a consequence.
The v2 server preflight verifies this exact two-field manifest diff before it
checks the live environment.

- Previous platform:
  `Linux-6.8.0-142-generic-x86_64-with-glibc2.39`
- Amended platform:
  `Linux-6.8.0-146-generic-x86_64-with-glibc2.39`
- Previous canonical observed-environment SHA-256:
  `985b3f67268b7c871807d0b36523439e766b663ef9ba33ed2bdc10c80f905524`
- Amended canonical observed-environment SHA-256:
  `6edd44c0dd3ae0757ad93fe136433342ef52527f2bc2b23be1416f73064d53c6`
- Previous environment-manifest file SHA-256:
  `c6d68d95a028793880b957edbee58ccb2cbca3005fd49690b89afe7a6815eec0`
- Amended environment-manifest file SHA-256:
  `8ec0986abbb0e345350a9c2d0f1fe644f3e6cc7dfc054bc5ac7724407dc363e0`

All other recorded environment fields are identical, including GPU identity
and memory, CUDA availability, cuDNN, PyTorch/CUDA, Python, package versions,
Mammoth commit, and runtime-patch declarations. The scientific hypotheses,
design, strength grid, seed sets, replica count, estimands, stopping rule, and
analysis plan are unchanged from v1.

## Scientific question and locked design

The outcome-informed follow-up asks at which *attainable replay support size*
reassigned replay ceases to outperform the same-seed conventional no-replay
reference. For a replay minibatch of 32, the authenticated historical 75%
request selects 24 rows and induces strength 23/31. The seven new requests
select every missing support size from 25 through 31 rows, inducing 24/31
through 30/31. The authenticated 100% endpoint selects 32 rows and induces
31/31.

New requested strengths (basis points):
`7742, 8065, 8387, 8710, 9032, 9355, 9677`.

- PMNIST: seeds 3801--3812, seven supports, four nested assignment replicas:
  336 new units.
- Split CIFAR-100: seeds 3501--3512, seven supports, four nested assignment
  replicas: 336 new units.
- Total: 672 new units. Every unit is mandatory; there is no optional stopping.
- Execution is serial on one GPU: PMNIST first, then CIFAR.

The primary result is the adjacent discrete selected-row/induced-strength
bracket containing the first raw positive-to-nonpositive mean contrast, issued
as unique only under the locked reversal rule. The non-increasing isotonic zero
is secondary, model-based, and reported only when its bracket agrees exactly
with the primary raw bracket. It must not be described as an exact percentage.

Inference uses the run seed as the independent unit, averages four assignment
replicas within seed/support, performs 20,000 whole-seed-profile bootstrap
replicates with seed 20261008, retains censored draws, and reports a grid-wise
max-deviation simultaneous band. CIFAR task-IL is primary and class-IL is a
mandatory separate secondary endpoint. PMNIST's fixed-current reference is a
separate sensitivity analysis.

## Final v2 source closure

- Source files: 42
- Crossover configuration SHA-256:
  `f134be622383afa8015c80295b1ede10febd7420e3580fb3b3f7c1b774e52cc1`
- Canonical source-bundle SHA-256:
  `b39f1df91d3209b9f04eb2c2ef194ad58242d90b4c4337c39deb785dfbabdd53`
- Source-manifest file SHA-256:
  `04f53c3bef89e486c1877ad02cd251563e4835727785de6a825b9a2ac19403dd`
- Source-archive SHA-256:
  `5f70eb390bccd9e3fd5ebf48d8d45b0b8d9cf68238b3db7e8b0fe5b428418510`

The archive was extracted into a fresh directory, contained exactly the 42
manifest files with no symbolic links or AppleDouble companions, passed
manifest reconstruction, and passed all 35 focused crossover tests. The full
local experiment test suite also exited successfully before sealing.

## External runtime pins

- Mammoth commit:
  `e75a491c69fd729edeb01431afb753d9157d9a81`
- Data manifest SHA-256:
  `5b2002ce8ea99481db0a6126090fe13797d456ad7a99a068e461c5d73177b23e`
- Amended environment manifest SHA-256:
  `8ec0986abbb0e345350a9c2d0f1fe644f3e6cc7dfc054bc5ac7724407dc363e0`
- Python executable target SHA-256:
  `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f`

These hashes are enforced by the scientific lock and/or server coordinator in
addition to data, environment, Mammoth, historical-anchor, and source
validation inside every queue and unit path.

## Authenticated historical anchors

PMNIST corrected-reference campaign:

- Parent lock: `100a1428f26860e0ef129b298192b05ccabc5c781ff1126b4528c5f4ddbcddec`
- Aggregate: `7d7cae877fc12e3b072b8c8a30ca0d3762947bf2f9ccea46dd09b99b5f575def`
- DONE: `1c97cb811bf84a843331ea6539a781b90af9fb7130ca140dabd25e36fd1cfc06`
- CAMPAIGN: `ed97e50e8e158771a455e87e6c64bffe8964219f2e3bd31550a9beade6f0c80e`

CIFAR assignment-strength campaign:

- Parent lock: `f8227a9f2248bd96bd88ddcf340acf7ae2514185cdbf80117e7ceebf596e808d`
- Aggregate: `da00d91c03a1954eaa10b53e34815f92b937cf66e9874d3e49c6e9b2f71ce4f8`
- DONE: `cb6ae04d1575580f2f983fa2fad97d65f3fbb0447854871537c3e6062c7931a0`

## V2 operator-layer hashes

- `LOCAL_TRANSFER.sh`:
  `af4bf7e636d042afbe0a9fd3082eaf85a09e83952b915425d9f54442ce992071`
- `LOCAL_FETCH.sh`:
  `77e1e2d79e1d4224e82ad9bccd3a275e3aeaee24062c71406c8f8fb993bf0972`
- `SERVER_INSTALL_PREFLIGHT.sh`:
  `22cd9c5b4e2bd3c92f926bb5ee5d0a421f94c3486e276e901f2a902c41b907a1`
- `SERVER_COORDINATOR.sh`:
  `011ef9929ee1b874e6e773e9f92bf019edd799ef2f87f3d9feecde043e033193`
- `SERVER_LAUNCH.sh`:
  `783c89f5a3280e70dd78a4bf14c20ea09c2f37c8c7a05a576bf9329c2b4e3e6a`
- `SERVER_MONITOR.sh`:
  `e93debc30a394a4ef7e4d60c72e52edce324515a6476c76c692e6013ad5485bc`
- `SERVER_PREPARE_FETCH.sh`:
  `9debf215c036f21b3fa12cea482204ceb3a14372134233353148e0a1fa7ce654`
- `OPERATOR_COMMANDS.md`:
  `acda627ec99a9a58506672be9b2f3f50cc1841ecc87e94d70130257cc6b19ac3`

The handoff checksum file binds the server-consumed source closure, both
environment manifests, all server scripts, and this seal. Result fetching is
allowed only after both 336-unit campaigns pass authenticated `--check-only`
validation.
