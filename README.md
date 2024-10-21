<p align="center">
  <h1 align="center">DepthSplat: Connecting Gaussian Splatting and Depth</h1>
  <p align="center">
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
    ·
    <a href="https://pengsongyou.github.io/">Songyou Peng</a>
    ·
    <a href="https://fangjinhuawang.github.io/">Fangjinhua Wang</a>
    ·
    <a href="https://hermannblum.net/">Hermann Blum</a>
    ·
    <a href="https://scholar.google.com/citations?user=U9-D8DYAAAAJ">Daniel Barath</a>
    ·
    <a href="http://www.cvlibs.net/">Andreas Geiger</a>
    ·
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2410.13862">Paper</a> | <a href="https://haofeixu.github.io/depthsplat/">Project Page</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://haofeixu.github.io/depthsplat/assets/teaser.png" alt="Logo" width="100%">
  </a>
</p>


<p align="center">
<strong>DepthSplat enables cross-task interactions between Gaussian splatting and depth estimation.</strong>
</p>



## Installation

Our code is developed based on pytorch 2.4.0, CUDA 12.4 and python 3.10. 

We recommend using [conda](https://docs.anaconda.com/miniconda/) for installation:

```bash
conda create -y -n depthsplat python=3.10
conda activate depthsplat

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.27.post2
pip install -r requirements.txt
```

## Model Zoo

Our models are hosted on Hugging Face 🤗 : https://huggingface.co/haofeixu/depthsplat

Model details can be found at [MODEL_ZOO.md](MODEL_ZOO.md).

We assume the downloaded weights are located in the `pretrained` directory.

## Camera Conventions

The camera intrinsic matrices are normalized (the first row is divided by image width, and the second row is divided by image height).

The camera extrinsic matrices are OpenCV-style camera-to-world matrices ( +X right, +Y down, +Z camera looks into the screen).

## Datasets

For RealEstate10K, please refer to [here](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) for acquiring the processed dataset.

For DL3DV, we plan to release our processed dataset in a few days.



## Depth Prediction

Please check [scripts/inference_depth_small.sh](scripts/inference_depth_small.sh), [scripts/inference_depth_base.sh](scripts/inference_depth_base.sh), and [scripts/inference_depth_large.sh](scripts/inference_depth_large.sh) for scale-consistent depth prediction with models of different sizes.

![depth](assets/depth.png)



We plan to release a simple depth inference pipeline in [UniMatch repo](https://github.com/autonomousvision/unimatch).



## Gaussian Splatting

- The training, evaluation, and rendering scripts on RealEstate10K dataset are available at [scripts/re10k_256x256_depthsplat_small.sh](scripts/re10k_256x256_depthsplat_small.sh), [scripts/re10k_256x256_depthsplat_base.sh](scripts/re10k_256x256_depthsplat_base.sh), and [scripts/re10k_256x256_depthsplat_large.sh](scripts/re10k_256x256_depthsplat_large.sh).

- The training, evaluation, and rendering scripts on DL3DV dataset are available at [scripts/dl3dv_256x448_depthsplat_base.sh](scripts/dl3dv_256x448_depthsplat_base.sh).

- Before training, you need to download the pre-trained [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [UniMatch](https://github.com/autonomousvision/unimatch) weights and set up your [wandb account](config/main.yaml) for logging.

```
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -P pretrained
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth -P pretrained
```


## Citation

```
@article{xu2024depthsplat,
      title   = {DepthSplat: Connecting Gaussian Splatting and Depth},
      author  = {Xu, Haofei and Peng, Songyou and Wang, Fangjinhua and Blum, Hermann and Barath, Daniel and Geiger, Andreas and Pollefeys, Marc},
      journal = {arXiv preprint arXiv:2410.13862},
      year    = {2024}
    }
```



## Acknowledgements

This project is developed with several fantastic repos: [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [UniMatch](https://github.com/autonomousvision/unimatch), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [DL3DV](https://github.com/DL3DV-10K/Dataset). We thank the original authors for their excellent work.

