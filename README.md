[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<p align="center">
  <a href="https://www.researchgate.net/publication/385525427_3D_Reconstruction_Tutorial_Data_Processing_Surface_Reconstruction_and_Texture_Mapping">
    <img src="https://user-images.githubusercontent.com/15963210/178556842-e1e7e1c0-206d-4960-82a5-3935c5cc60b4.png" alt="Logo" width="150"><br>
    <b><span style="font-size: 28px;">
      3D Reconstruction Tutorial: Data Processing, Surface Reconstruction, and Texture Mapping
    </span></b>
  </a>
</p>

![biblioteca-framed](https://github.com/user-attachments/assets/f62e4336-a06d-4637-9853-b6efd357ceb1)

## Setup

Requires Conda.

```bash
git clone https://github.com/tiagomfmadeira/3dr-tutorial.git
cd 3dr-tutorial
conda create -n 3dr_tutorial python=3.9 pip -y
conda activate 3dr_tutorial
python -m pip install .
```

## Tutorial data

Download the [tutorial data](https://doi.org/10.5281/zenodo.23805485)
and extract it into `src/`, producing `src/cache/camera_data.pickle`.

## Run

```bash
cd src
python -m notebook
```

Open `3DR_Tutorial.ipynb` and run the cells in order. In VS Code, select the
`3dr_tutorial` environment as the notebook kernel. Generated meshes and textures
are written to `src/output/`.

## License

Code: [GPL-3.0](LICENSE). Tutorial data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
