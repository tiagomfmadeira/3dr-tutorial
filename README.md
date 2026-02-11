<p align="center">
  <a href="https://www.researchgate.net/publication/385525427_3D_Reconstruction_Tutorial_Data_Processing_Surface_Reconstruction_and_Texture_Mapping">
    <img src="https://user-images.githubusercontent.com/15963210/178556842-e1e7e1c0-206d-4960-82a5-3935c5cc60b4.png" alt="Logo" width="150"><br>
    <b><span style="font-size: 28px;">
      3D Reconstruction Tutorial: Data Processing, Surface Reconstruction, and Texture Mapping
    </span></b>
  </a>
</p>

![biblioteca-framed](https://github.com/user-attachments/assets/f62e4336-a06d-4637-9853-b6efd357ceb1)

---

It is recommended that you use Conda to create a python virtual environment so it is easier to manage dependencies and cleanup.\
To install Conda, head over to [Installing conda — conda 24.7.2.dev87 documentation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)\
Miniconda will suffice, but feel free to install the Anaconda Distribution.

---

Once Conda has been successfully installed, open the prompt and let's create a python virtual environment (venv):

```bash
conda create -n 3dr_tutorial python==3.9
```

Once it is created, we must activate the environment:

```bash
conda activate 3dr_tutorial
```
 
 You should now see **(3dr_tutorial)** as a start of line on the cli instead of **(base)**. †

---

Download the contents of this folder: [3dr_tutorial](https://uapt33090-my.sharepoint.com/:f:/g/personal/tiagomadeira_ua_pt/EpT8T-TXWaFBjAZkjcvxr7kB5Zb2afHl2z1BUuYnCr5qRg?e=SsoUzl)
to your desired working directory.\
Let's now install all the required dependencies for our project.

```bash
cd <working_dir>
pip install -r requirements.txt
```

We are now ready to open up our jupyter notebook.

```bash
jupyter notebook
```

---

† If at any point you would like to remove this environment and start over, you can do so by deactivating and removing it using the following commands:

```bash
conda deactivate 3dr_tutorial
conda remove -n 3dr_tutorial -all
```

---

## License

Distributed under the **GPL-3.0 License**. See [`LICENSE`](LICENSE) for more information.
