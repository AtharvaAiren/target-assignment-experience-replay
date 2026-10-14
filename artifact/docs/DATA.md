# Data

No raw third-party data or pretrained weights are redistributed.

Do not commit datasets, caches, or model weights. The correct location depends
on the campaign: some runners accept an explicit path, while the older shallow
and breadth loaders use a fixed repository-relative cache. The paths below are
the paths the shipped code actually reads.

## Shallow regression streams

The METRO, BEIJING, APPLIANCES, and TETOUAN loaders download public UCI
archives when their fixed cache files are absent:

| Dataset | Source | Cache path |
|---|---|---|
| Metro Interstate Traffic Volume | `https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip` | `data_raw/uci/metro.zip` |
| Beijing Multi-Site Air Quality | `https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip` | `data_raw/uci/beijing.zip` |
| Appliances Energy Prediction | `https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip` | `data_raw/uci/appliances_workshop.zip` |
| Tetouan City Power Consumption | `https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip` | `data_raw/uci/tetouan_workshop.zip` |

These paths are not configurable command-line flags in the shipped shallow
loaders. Run from the repository root and leave the cache untracked.

## CNN/GRU breadth

The classification breadth runner reads CIFAR-100 at
`data_raw/cifar100/`, as fixed in `configs/breadth.json`; torchvision may
download it there. The sequence breadth runner uses the Appliances archive
above and verifies the locked SHA-256 from the configuration. There is no
generic `--data-root` flag for this campaign.

## Continual-learning campaigns

The original mechanism runner accepts `--data-root`. Prepare an external
absolute directory with this exact top-level layout:

```text
<data-root>/
  PUBLIC_DATA_ACQUISITION.json
  CIFAR100/
  MNIST/
  MINIIMG/
    MINIIMAGENET_MANIFEST.json
    ... local miniImageNet split files and images ...
```

Download and validate the public datasets with:

```bash
PYTHONPATH=.:src python -m experiments.gradient_reuse_mechanisms.acquire_public \
  --data-root /absolute/path/to/mechanism-data
```

miniImageNet is not downloaded by that command. Supply the canonical
`train.csv`, `val.csv`, and `test.csv` files plus the local `images/` tree, then
inspect the exact preparation interface:

```bash
PYTHONPATH=.:src python -m experiments.gradient_reuse_mechanisms.miniimagenet --help
```

The optional Hugging Face conversion route uses gated ImageNet access and is
documented by:

```bash
PYTHONPATH=.:src python -m experiments.gradient_reuse_mechanisms.miniimagenet_hf --help
```

Finally, freeze all acquired bytes into a manifest stored **outside** the data
root:

```bash
PYTHONPATH=.:src python -m experiments.gradient_reuse_mechanisms.data_manifest \
  --data-root /absolute/path/to/mechanism-data \
  --output /absolute/path/to/locks/MECHANISM_DATA_MANIFEST.json
```

The campaign then receives both `--data-root` and `--data-manifest`. The
Mammoth checkout is a separate path and must remain disjoint from both.

## Finite-horizon Beijing analysis

Pass the downloaded Beijing archive directly with `--data-archive`. The full
run accepts only the archive and derived-array identity fixed in
`configs/finite_horizon_beijing.json` (archive SHA-256
`b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`).
The source URL is the Beijing UCI URL in the first table.

## Transformer assets

`configs/transformer.json` locks both upstream repositories and full revisions:

- model: `EleutherAI/pythia-160m-deduped`, revision
  `de95c9570237bac1e0c70856f41b840809c55889`;
- dataset: `common-pile/news`, revision
  `13e76bdc8d49ed14d710fac7e3b61186cf74c8d3`.

Acquire and materialize those snapshots into a new external asset root with:

```bash
PYTHONPATH=.:src python -m experiments.gradient_reuse_transformer.acquire \
  --config configs/transformer.json \
  --source-manifest provenance/source/transformer.json \
  --output-root /absolute/path/to/transformer-assets
```

The scientific runner is offline-only and validates
`/absolute/path/to/transformer-assets/ASSET_MANIFEST.json` plus the current
asset bytes. Hugging Face access may require accepting the upstream terms and
authenticating locally. The amended reviewer-extension Pythia campaign also
requires a historical correction/parent chain that is not fully distributed;
its frozen result is verifiable, but it is not a self-contained live rerun in
this release.
