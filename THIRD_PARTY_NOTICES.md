# Third-party notices and non-code assets

The root MIT license applies to original software in this repository. The
following components have separate attribution or terms.

## IFRNet

`src/tbfe/baselines/extrass.py` adapts the multiscale encoder/decoder design
from [IFRNet](https://github.com/ltkong218/IFRNet).

MIT License

Copyright (c) 2022 Lingtong Kong

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

No third-party implementation of Softmax Splatting, ARFlow, EdgeConnect, or
StructureFlow is included in this release.

## Factory smoke fixture

`examples/factory_smoke/frame.npz` is a 128 x 128 input-only crop prepared by
the paper authors. The authors permit this unmodified fixture to be publicly
redistributed with this repository and used solely to verify code execution.
It is not covered by the software MIT license and is not licensed as training,
evaluation, or benchmark data. Copyright (c) 2026 the Tile-based Frame
Extrapolation authors. All rights not expressly granted are reserved.

## Placeholder BRDF lookup table

`src/tbfe/preprocess/assets/Precomputed.exr` is an author-provided placeholder
EXR included only to exercise asset loading and preprocessing plumbing. It is
not intended as a paper-reproduction asset; users should supply lookup data
appropriate to their renderer for research runs. The authors permit this file
to be redistributed with the repository. It is not covered by the software
MIT license. Its SHA256 is
`aafbd6c78fe7974159af02c7cc9be7999949457bacff8352aa984f0db3fa761a`.
