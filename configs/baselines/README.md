# Baselines and warp ablation

The architectures are available from `tbfe.baselines`:

- `ExtraNet`: reported recurrent extrapolation baseline;
- `ExtraSSNet`: reported ExtraSS baseline;
- `ExtraSSTbrNet`: the ExtraSS architecture driven by TBR warp results.

Their interfaces are explicit tensor APIs documented in the class docstrings.
No baseline weights or private scene paths are distributed. `ExtraSSTbrNet`
does not perform the experiment code's ground-truth sky replacement.
