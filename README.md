# Tile-based Frame Extrapolation

Official PyTorch implementation of **Tile-based Frame Extrapolation**, accepted
to SIGGRAPH Asia 2026.

Liang Pu, Haodong Tian, Jiawei Zhang, Weitao Zhang, Yanwen Guo, Junqiu Zhu, and
Jie Guo (corresponding author).

Repository: <https://github.com/NJUCG/tile-based-frame-extrapolation>

> Paper and project-page links will be added when they become public.

## What is released

This repository contains the method code rather than an experiment snapshot:

- the G-buffer-guided TBR warp and its OpenGL compute shader;
- the dual-encoder tiled irradiance-repair network;
- tile selection, context expansion, batching, and overlap-aware reconstruction;
- path-agnostic training, inference, evaluation, benchmarking, and ONNX-export
  scaffolding;
- the ExtraNet and ExtraSS baselines, plus the ExtraSS-TBR ablation;
- a small **input-only** Factory crop for automated smoke testing.

Training data and pretrained weights are not distributed. The trained repair
network is scene-dependent; users should train it on buffers exported from
their own rendering setup. The included Factory crop has no ground truth and
is not a benchmark or qualitative demo.

The packaged `Precomputed.exr` is a placeholder that exercises asset loading
and preprocessing plumbing. It is not a paper-reproduction asset; research
runs should supply lookup data appropriate to their renderer.

## Method at a glance

1. TBR warps the previous irradiance with motion and G-buffer guidance.
2. The invalid-region mask selects only tiles containing enough missing pixels.
3. Selected tiles receive additional spatial context and pass through the
   repair network together with current-frame G-buffers.
4. Predicted residual tiles are averaged in overlap regions and added only
   inside the selected invalid mask.
5. Repaired irradiance is multiplied by current-frame albedo to recover color.

The public inference path never reads ground-truth `Color`, `Irradiance`,
`SkyMask`, or `ShadowMask`.

## Release safety

Only the contents of this repository should be published; the internal
research tree is not part of the release. Before every public commit or tag,
run:

```bash
python tools/audit_release.py
```

The audit rejects likely credentials, private machine paths, SSH-transfer
helpers, model/data artifacts, unreviewed binaries, oversized files, and
suspicious Git history. It also pins the exact hashes of the two deliberately
included test assets. The same audit runs as the first CI check.

## Environment

The release was tested on Ubuntu 22.04, Python 3.11, PyTorch 2.9.1 + CUDA 12.6,
and an NVIDIA RTX 4090. CPU execution is supported.

```bash
conda create -n tbfe python=3.11 -y
conda activate tbfe

# CUDA 12.6; use the corresponding official PyTorch command for another target.
pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
```

For raw EXR and OpenGL preprocessing:

```bash
pip install -e ".[preprocess]"
```

## Quick smoke test

No checkpoint is required for this plumbing test:

```bash
python tools/infer.py \
  --input examples/factory_smoke/frame.npz \
  --random-init \
  --device cpu \
  --output outputs/factory_smoke.npz

pytest -q
```

`--random-init` is deliberately explicit: the output verifies buffer loading,
tile selection, network execution, and reconstruction, but has no visual
meaning.

## Inference with your checkpoint

The required processed inputs are:

| Buffer | Channels | Meaning |
| --- | ---: | --- |
| `TbrIrradiance_1` | 3 | TBR-warped irradiance |
| `WarpToCurrGbufferMask_1` | 1 | one means invalid/needs repair |
| `Depth` | 1 | current-frame depth |
| `Normal` | 3 | current-frame normal |
| `Metallic` | 1 | current-frame metallic value |
| `Roughness` | 1 | current-frame roughness |
| `Albedo` | 3 | current-frame BRDF albedo |

Store each frame as `<Buffer>.<frame-id>.npy` in HWC layout, then run:

```bash
python tools/infer.py \
  --input /path/to/processed/sequence \
  --frame 0007 \
  --checkpoint /path/to/weights.pt \
  --output outputs/prediction.npz
```

Plain state dictionaries, Lightning `.ckpt` files, and safetensors are
supported. PyTorch checkpoints are pickle-based; load only trusted files.

## Training

Edit `data.train_sequences` and, optionally, `data.val_sequences` in
[`configs/train.yaml`](configs/train.yaml), then run:

```bash
python tools/train.py --config configs/train.yaml
```

The trainer writes plain model state dictionaries. Dataset splits are not
included because the paper dataset is not distributed.

## TBR preprocessing

The reusable API is under `tbfe.preprocess`. A single TBR warp can be executed
from an `.npz` containing the seven fields of `TBRWarpInputs`:

```bash
python tools/warp_tbr.py --input warp_inputs.npz --output TbrIrradiance_1.npy
```

See [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md) for conventions and
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for the processed format. OpenGL
4.5 and EGL are required for the packaged headless implementation.

## Evaluation

Evaluation never injects ground-truth pixels into saved predictions. An
optional valid-region mask can be supplied and must be shared by every method:

```bash
python tools/evaluate.py \
  --prediction outputs/prediction.npz \
  --target /path/to/target.npz \
  --valid-mask /path/to/valid_mask.npy
```

The historical paper pipeline replaced sky and shadow regions with ground
truth only for metric computation. The release expresses any such protocol as
an explicit metric mask instead, keeping inference ground-truth-free.

## Baselines, ablations, and deployment

- `tbfe.baselines.ExtraNet` and `ExtraSSNet` are the reported baselines.
- `ExtraSSTbrNet` exposes the TBR-warp ablation without GT sky replacement.
- `--tile-size`, `--tile-expand`, and `--min-ratio` expose the tile ablations.
- `tools/export_onnx.py` exports the per-tile network kernel; selection and
  merging remain host-side.
- `tools/benchmark.py` reports the full tiled path with fixed warm-up and
  synchronization. A random model is acceptable only for runtime plumbing.

Following the scope used by similar rendered-frame-prediction releases, the
ablation mechanisms are public, but per-ablation checkpoints and table-specific
runner scripts are not provided.

## Repository layout

```text
configs/                 path-free training and ablation settings
docs/                    data, preprocessing, and reproducibility notes
examples/factory_smoke/  small input-only code fixture
src/tbfe/models/         main repair network
src/tbfe/ops/            tile selection/extraction/merge
src/tbfe/preprocess/     TBR warp and buffer utilities
src/tbfe/baselines/      ExtraNet, ExtraSS, and ExtraSS-TBR
tools/                   train/infer/evaluate/export/benchmark entry points
tests/                   CPU unit and smoke tests
```

## Citation

```bibtex
@inproceedings{pu2026tilebased,
  title     = {Tile-based Frame Extrapolation},
  author    = {Pu, Liang and Tian, Haodong and Zhang, Jiawei and
               Zhang, Weitao and Guo, Yanwen and Zhu, Junqiu and Guo, Jie},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026}
}
```

Please replace this provisional entry with the ACM BibTeX once the paper DOI
is available.

## License and contact

Original software is released under the MIT License. Third-party and non-code
assets are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Questions: Jie Guo, [guojie@nju.edu.cn](mailto:guojie@nju.edu.cn).

Before making the repository public, complete
[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).
