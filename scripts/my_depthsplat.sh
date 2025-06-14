#!/usr/bin/env bash


# base model
# first train on ARKitScenes, 2 views, 256x448
# train on 8x (8 nodes) 4x GPUs (>=80GB VRAM) for 150K steps, batch size 8 on each gpu 
python -m src.main +experiment=scannet_plus \
data_loader.train.batch_size=8 \
dataset.test_chunk_interval=10 \
trainer.max_steps=1500 \
trainer.num_nodes=1 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feat \
ure_resolution=1 
model.encoder.monodepth_vit_type=vits \
checkpointing.pretrained_monodepth=pretrained/depth-anything/model_s.ckpt \
output_dir=checkpoints/train

python -m src.main +experiment=scannet_plus data_loader.train.batch_size=8 dataset.test_chunk_interval=10 trainer.max_steps=1500 trainer.num_nodes=1 model.encoder.num_scales=2 model.encoder.upsample_factor=4 model.encoder.lowest_feat ure_resolution=1model.encoder.monodepth_vit_type=vits checkpointing.pretrained_monodepth=pretrained/depth-anything/model_s.ckpt output_dir=checkpoints/train


# test
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=scannet_plus \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
dataset.image_shape=[256,256] \
model.encoder.monodepth_vit_type=vits \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
checkpointing.pretrained_monodepth=pretrained/depth-anything/model_s.ckpt \
mode=test \
dataset/view_sampler=evaluation \


python -m src.main +experiment=scannet_plus dataset.test_chunk_interval=1 model.encoder.num_scales=2 model.encoder.upsample_factor=2 model.encoder.lowest_feature_resolution=4 model.encoder.monodepth_vit_type=vits checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth checkpointing.pretrained_monodepth=pretrained/depth-anything/model_s.ckpt mode=test dataset/view_sampler=bounded