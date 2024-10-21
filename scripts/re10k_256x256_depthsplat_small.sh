#!/usr/bin/env bash


# small model
# train on 8x GPUs, batch size 4 on each gpu (>=32GB memory)
python -m src.main +experiment=re10k \
data_loader.train.batch_size=4 \
dataset.test_chunk_interval=10 \
trainer.val_check_interval=0.5 \
trainer.max_steps=150000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_regressor_channels=16 \
model.encoder.feature_upsampler_channels=64 \
model.encoder.return_depth=true \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
wandb.project=depthsplat \
output_dir=checkpoints/re10k-depthsplat-small


# evaluate on re10k
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=1 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_regressor_channels=16 \
model.encoder.feature_upsampler_channels=64 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-49b2d15c.pth \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.mode=disabled \
test.save_image=false \
test.save_gt_image=false \
output_dir=output/tmp


# render video on re10k (need to have ffmpeg installed)
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset.test_chunk_interval=10 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_regressor_channels=16 \
model.encoder.feature_upsampler_channels=64 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-49b2d15c.pth \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.compute_scores=false \
wandb.mode=disabled \
test.save_image=false \
test.save_gt_image=false \
output_dir=output/tmp

