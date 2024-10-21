#!/usr/bin/env bash


# large model: depth prediction on re10k 352x640
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
data_loader.train.batch_size=1 \
dataset.test_chunk_interval=10 \
mode=test \
dataset.near=0.5 \
dataset.far=100. \
dataset/view_sampler=evaluation \
dataset.image_shape=[352,640] \
test.compute_scores=false \
dataset.view_sampler.num_context_views=2 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitl \
model.encoder.return_depth=true \
train.forward_depth_only=true \
checkpointing.pretrained_depth=pretrained/depthsplat-depth-large-50d3d7cf.pth \
wandb.project=depthsplat \
wandb.mode=disabled \
test.save_image=false \
test.save_depth=true \
test.save_depth_concat_img=true \
output_dir=output/depthsplat-depth-large-re10k


# large model: depth prediction on dl3dv 256x480
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
data_loader.train.batch_size=1 \
dataset.test_chunk_interval=1 \
mode=test \
dataset.near=0.5 \
dataset.far=100. \
dataset.roots=[datasets/dl3dv] \
dataset/view_sampler=evaluation \
dataset.image_shape=[256,480] \
test.compute_scores=false \
dataset.view_sampler.num_context_views=2 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitl \
model.encoder.return_depth=true \
train.forward_depth_only=true \
checkpointing.pretrained_depth=pretrained/depthsplat-depth-large-50d3d7cf.pth \
wandb.project=depthsplat \
wandb.mode=disabled \
test.save_image=false \
test.save_depth=true \
test.save_depth_concat_img=true \
output_dir=output/depthsplat-depth-large-dl3dv

