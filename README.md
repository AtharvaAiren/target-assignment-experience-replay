# How Target Assignment Shapes Experience Replay

**Atharva Airen and Anirban Pal** · Extended manuscript under review at Transactions on Machine Learning Research (TMLR).

Which benefits of experience replay depend on preserving the original input–target assignment? The artifact includes replay interventions, locked configurations, compact evidence and paper-value checks.

[Portfolio](https://github.com/AtharvaAiren) · [Code and reproduction guide](artifact/README.md) · [Related version](https://github.com/AtharvaAiren/gradient-reuse-newinml-2026)

## Start with the code

The original experimental release is preserved in **[artifact/](artifact/)**, including its original checksum manifest. The portfolio description lives outside that frozen directory, so published results and source hashes remain intact.

For CPU-only checks of the retained evidence:

```bash
cd artifact
python scripts/verify_results.py
python scripts/verify_paper_values.py
```

Use the artifact's reproduction guide for installation and full experiment requirements. These checks inspect stored evidence; they do not rerun model training. External datasets, model weights and feature stores are not bundled.

## Version and attribution

This repository is one version of a research work, not an additional independent paper. The related-version link above distinguishes the workshop study from the extended manuscript. Historical titles and anonymous citation metadata may remain inside the frozen release.

The artifact's original license and third-party notices apply to its contents. Its review-release terms are retained without replacing them with a new license.
