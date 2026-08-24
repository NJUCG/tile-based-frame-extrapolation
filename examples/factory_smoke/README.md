# Factory smoke fixture

`frame.npz` is an input-only 128 x 128 crop from `Factory/test1_30`, frame
`0007`. It contains the seven public inference buffers plus four provenance
fields recording the scene, sequence, frame, and crop coordinates. It contains
no ground truth. The crop was selected to contain invalid pixels so tile
selection is exercised.

Run it with `tools/infer.py --random-init` to verify code execution. The
resulting image has no qualitative meaning and this fixture is not a dataset or
benchmark.

Public redistribution is permitted only for this smoke-testing purpose; see
`THIRD_PARTY_NOTICES.md` at the repository root for the complete terms.
