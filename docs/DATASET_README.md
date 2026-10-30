# 3D Reconstruction Tutorial: Camera Cache

Version 1.0.0 · CC BY 4.0

Cached camera data for the [3D reconstruction tutorial](https://github.com/tiagomfmadeira/3dr-tutorial).
The cache is the notebook's intended input and avoids preprocessing during the tutorial.

## Use

1. Clone the code repository and follow its environment setup.
2. Extract this archive into `src/`, producing `src/cache/camera_data.pickle`.
3. From `src/`, run `python -m notebook` and open `3DR_Tutorial.ipynb`.

The notebook generates its outputs in `src/output/`. Code and generated meshes
are not included in this archive. The original E57 is not required for this workflow.
Only load pickle files from a trusted source; loading them can execute code.

## Attribution and license

Tiago Madeira, Lucas Dal'Col, Miguel Riem Oliveira, and Paulo Dias.
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Related tutorial: https://doi.org/10.13140/RG.2.2.18662.54086

`SHA256SUMS.txt` contains the cache checksum. The cache is distributed unchanged
from the original tutorial materials.
