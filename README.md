# BiVSS-CD

Official implementation of **BiVSS-CD: Bidirectional Semantic Object
Propagation for Training-Free Open-Vocabulary Remote Sensing Change
Detection**.

BiVSS-CD converts a bi-temporal image pair into forward and backward
pseudo-videos. SAM3 uses category text prompts to segment and propagate semantic
objects. Semantic-Spatial Cross-Comparison and Elimination (SSCCE) suppresses
mask deformation, registration noise, and object-ID drift, and bidirectional
consensus retains changes confirmed from both temporal directions.

> The paper has been submitted. A bibliographic record will be added after it
> becomes publicly available.

## Installation

Requirements: Python 3.10+, a CUDA-capable PyTorch environment supported by
SAM3, and Git.

```bash
git clone --recurse-submodules https://github.com/KimotaQY/BiVSS-CD.git
cd BiVSS-CD
pip install -e .
pip install -e third_party/sam3
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The submodule is pinned to the BiVSS-CD SAM3 fork because the method requires
the fork's decoupled object-selection interface. Do not replace it with an
arbitrary SAM3 checkout. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
At startup, BiVSS-CD verifies that Python actually imports `sam3` from this
submodule and fails with the imported and expected paths if another editable or
site-packages installation shadows it.

## Model weights

Obtain the SAM3 checkpoint under Meta's terms by following the instructions in
the SAM3 submodule. Checkpoints are not redistributed here. Set `checkpoint` in
a local YAML file or pass `--checkpoint` on the command line. Do not commit
weights to this repository.

## Quick inference

Choose the dataset-specific configuration, then run:

```bash
python -m bivss_cd.infer \
  --image-t1 examples/t1.png \
  --image-t2 examples/t2.png \
  --output outputs/change.png \
  --config configs/whu_cd.yaml \
  --checkpoint /path/to/sam3.pt
```

Multiple open-vocabulary categories can be supplied after `--prompts`.
For exact paper reproduction, all prompt masks from both temporal directions
vote jointly and pixels supported by at least two votes are retained.

```python
from bivss_cd import BiVSSCD, BiVSSConfig

config = BiVSSConfig.from_yaml("configs/whu_cd.yaml", checkpoint="/path/to/sam3.pt")
result = BiVSSCD(config).predict("t1.png", "t2.png", config.prompts)
binary_change = result.binary_mask
building_change = result.class_masks["building"]
```

The two images must have identical dimensions. `ChangeResult.binary_mask` is a
two-dimensional `uint8` array containing 0 and 1. Per-prompt masks use the same
format.

## Dataset evaluation

Datasets are not redistributed. Download them from their official project
pages and arrange their published test splits as follows:

```text
LEVIR-CD or WHU-CD
└── test
    ├── A
    ├── B
    └── label

SECOND
└── test
    ├── im1
    ├── im2
    ├── label
    ├── building_label
    ├── ground_label
    ├── low_vegetation_label
    ├── playground_label
    ├── tree_label
    └── water_label
```

Binary building-change evaluation:

```bash
python -m bivss_cd.evaluate \
  --dataset LEVIR-CD \
  --data-root /path/to/LEVIR-CD \
  --output-dir outputs/levir \
  --config configs/levir_cd.yaml \
  --checkpoint /path/to/sam3.pt
```

Each SECOND category must be evaluated with its matching configuration because
the paper uses different thresholds for different semantic categories. The
category outputs can subsequently be combined into a semantic change map; class
ID 0 is background/no-change.

The category adapters select labels automatically: `SECOND_building` uses
`building_label`, `SECOND_surface` uses `ground_label`, `SECOND_low_veg` uses
`low_vegetation_label`, and the playground, tree, and water variants use their
correspondingly named label directories.

Evaluation writes predictions, `per_sample.csv`, `summary.json`, and
`evaluation.log`. Re-running the same command skips samples already present in
the CSV. Binary results include IoU, F1, precision, recall, and overall accuracy;
SECOND additionally reports per-class and mean IoU/F1 when semantic labels are
available.

The metrics in `summary.json` are computed from one confusion matrix accumulated
over every evaluated pixel, matching the paper's official folder evaluator.
They are not the arithmetic mean of the per-image metrics in `per_sample.csv`.
After changing an algorithm version, pass `--force` or use a new output
directory so predictions from an older run are not reused.
The evaluator writes `run_manifest.json` and refuses to resume when the
BiVSS-CD version, dataset, prompts, or configuration differs from the existing
run.

## Configuration

The repository contains the following paper configurations:

| Configuration | Prompt(s) | IoU | Detection | New object | Decoupled |
| --- | --- | ---: | ---: | ---: | --- |
| `whu_cd.yaml` | `roof` | 0.30 | 0.55 | 0.60 | false |
| `levir_cd.yaml` | `roof` | 0.10 | 0.10 | 0.15 | false |
| `levir_cd_512.yaml` | `roof` | 0.30 | 0.35 | 0.40 | false |
| `second_building.yaml` | `building` | 0.10 | 0.30 | 0.40 | false |
| `second_tree.yaml` | `tree`, `forest` | 0.10 | 0.15 | 0.20 | false |
| `second_water.yaml` | `water,river,pond,sea` | 0.50 | 0.10 | 0.20 | false |
| `second_low_veg.yaml` | `low vegetation`, `grass` | 0.10 | 0.20 | 0.30 | false |
| `second_surface.yaml` | `the ground`, `bare land` | 0.40 | 0.40 | 0.45 | false |
| `second_playground.yaml` | three court prompts | 0.40 | 0.35 | 0.50 | false |

Prompts are loaded from YAML unless `--prompts` explicitly overrides them.
Important parameters:

- `iou_threshold`: SSCCE morphology/context threshold.
- `score_threshold_detection`: SAM3 detection threshold.
- `new_det_thresh`: threshold for introducing a new tracked object.
- `use_decoupled_selection`: use independent tracker states for new objects.
- `gpus_to_use`: `all` reproduces `test_script.py`; use a YAML list such as
  `[0]` to restrict inference to selected GPUs.
- `consensus`: `intersection` for the paper method or `union` for analysis.

The default inference backend requires CUDA. CPU mode exists only for unit tests
that inject a lightweight predictor.

## Development and verification

Install test dependencies and run:

```bash
pip install -e ".[test]"
pytest
python tools/release_check.py
```

Unit tests do not download datasets or model weights. End-to-end SAM3 inference
requires the checkpoint and a CUDA environment.

## Troubleshooting

- **SAM3 is not importable:** initialize the submodule and install it editable.
- **Required SAM3 interface is missing:** restore the pinned submodule revision.
- **Checkpoint not found:** use an absolute local checkpoint path in an untracked
  config or pass `--checkpoint`.
- **CUDA memory pressure:** evaluate fewer samples with `--limit`; each prompt is
  processed in a lifecycle-safe SAM3 session.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). The final IEEE
BibTeX entry will be added after publication.

## License

BiVSS-CD's original code is licensed under Apache-2.0. SAM3 code and weights are
subject to the separate SAM License; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
