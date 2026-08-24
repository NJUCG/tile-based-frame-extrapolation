# Data card

The training and evaluation dataset used by the paper is not distributed.

`examples/factory_smoke/frame.npz` is the only included renderer-frame fixture.
It is a 128 x 128 crop with the seven inference inputs and four disclosed
provenance fields (scene, sequence, frame, and crop coordinates). It contains
no target image and exists solely for automated code validation. It must not
be used to report quality metrics. The separately packaged `Precomputed.exr`
is a placeholder lookup asset, not renderer-frame data.

Users are responsible for licensing renderer scenes and exported buffers used
to train their own models.
