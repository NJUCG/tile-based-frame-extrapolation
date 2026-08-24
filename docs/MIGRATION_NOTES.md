# Research-code to release-code mapping

| Research workspace | Public implementation |
| --- | --- |
| `model/ImageInpaintNet.py` | `src/tbfe/models/inpainting.py` |
| inline `_make_tiled_batch_v3` and fold helpers | `src/tbfe/ops/tiling.py` |
| `ImageInpaintNetLoss` in the shared loss registry | `src/tbfe/losses.py` |
| internal dataset adapter | `src/tbfe/data.py` |
| `preprocess/shader/tbr/GGWarp.comp` | `src/tbfe/preprocess/shaders/GGWarp.comp` |
| TBR draw pass and numerical preprocessing | `src/tbfe/preprocess/` |
| `model/ExtraNet.py` | `src/tbfe/baselines/extranet.py` |
| `model/ExtraSSNet.py` | `src/tbfe/baselines/extrass.py` |
| Lightning/Hydra experiment wrapper | explicit scripts under `tools/` |

The public model separates inference inputs from training targets. It does not
carry forward the research evaluator's sky/shadow ground-truth compositing.
