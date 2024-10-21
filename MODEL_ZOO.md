# Model Zoo

We provide models for scale-consistent depth estimation and view synthesis with 3D Gaussian splatting.



## Scale-Consistent Depth Estimation

- The depth models are trained with the following procedure:
  - Initialize the monocular feature with Depth Anything V2 and the multi-view Transformer with UniMatch.
  - Train the full DepthSplat model end-to-end on the RealEstate10K dataset.
  - Fine-tune the pre-trained depth model on the depth datasets with ground truth depth supervision. The depth datasets used for fine-tuning include ScanNet, DeMoN, TartanAir, and VKITTI2.
- All the depth models are fine-tuned with two images as input, the training image resolution is 352x640.
- The scale of the predicted depth is aligned with the scale of camera pose's translation.

| Model                  | Monocular | Multi-View | Params (M) |                           Download                           |
| ---------------------- | :-------: | :--------: | :--------: | :----------------------------------------------------------: |
| depthsplat-depth-small |   ViT-S   |  1-scale   |     36     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-depth-small-3d79dd5e.pth) |
| depthsplat-depth-base  |   ViT-B   |  2-scale   |    111     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-depth-base-f57113bd.pth) |
| depthsplat-depth-large |   ViT-L   |  2-scale   |    338     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-depth-large-50d3d7cf.pth) |



## Gaussian Splatting

- The models are trained on RealEstate10K and/or DL3DV datasets at 256x256 or 256x448 resolutions.
- We plan to release more high-resolution models in the future.

| Model                             | Monocular | Multi-View | Params (M) |                           Download                           |
| --------------------------------- | :-------: | :--------: | :--------: | :----------------------------------------------------------: |
| depthsplat-gs-small-re10k-256x256 |   ViT-S   |  1-scale   |     37     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-small-re10k-256x256-49b2d15c.pth) |
| depthsplat-gs-base-re10k-256x256  |   ViT-B   |  2-scale   |    117     | [download](https://huggingface.co/haofeixu/depthsplat/blob/main/depthsplat-gs-base-re10k-256x256-044fdb17.pth) |
| depthsplat-gs-large-re10k-256x256 |   ViT-L   |  2-scale   |    360     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-large-re10k-256x256-288d9b26.pth) |
| depthsplat-gs-base-dl3dv-256x448  |   ViT-B   |  2-scale   |    360     | [download](https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-dl3dv-256x448-75cc0183.pth) |

