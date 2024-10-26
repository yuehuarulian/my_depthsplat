#!/usr/bin/env bash


# base model
# first train on re10k, view 2
# 8x GPUs, batch size 1 on each gpu (>=14GB memory)
python -m src.main +experiment=re10k \
data_loader.train.batch_size=1 \
dataset.test_chunk_interval=10 \
trainer.val_check_interval=0.5 \
dataset.far=200. \
dataset.image_shape=[256,448] \
trainer.max_steps=100000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_regressor_channels=32 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=128 \
model.encoder.return_depth=true \
checkpointing.pretrained_depth=pretrained/depthsplat-depth-base-f57113bd.pth \
wandb.project=depthsplat \
output_dir=checkpoints/re10k-256x448-depthsplat-base


# finetune on dl3dv, random view 2-6
# 4x GPUs, batch size 1 on each gpu (>=43GB memory)
# resume from the previously pretrained model on re10k
PRETRAINED_MODEL_ON_RE10K=xxx.pth

python -m src.main +experiment=dl3dv \
data_loader.train.batch_size=1 \
dataset.test_chunk_interval=1 \
trainer.val_check_interval=0.2 \
train.eval_model_every_n_val=3 \
dataset.roots=[datasets/dl3dv] \
dataset.near=1. \
dataset.far=200. \
dataset.view_sampler.num_target_views=8 \
dataset.view_sampler.num_context_views=6 \
dataset.min_views=2 \
dataset.max_views=6 \
dataset.view_sampler.min_distance_between_context_views=20 \
dataset.view_sampler.max_distance_between_context_views=50 \
dataset.view_sampler.context_gap_warm_up_steps=10000 \
dataset.view_sampler.initial_min_distance_between_context_views=15 \
dataset.view_sampler.initial_max_distance_between_context_views=30 \
trainer.max_steps=100000 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_regressor_channels=32 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=128 \
model.encoder.multiview_trans_nearest_n_views=3 \
model.encoder.costvolume_nearest_n_views=3 \
train.train_ignore_large_loss=0.1 \
model.encoder.return_depth=true \
checkpointing.pretrained_model=${PRETRAINED_MODEL_ON_RE10K} \
checkpointing.every_n_train_steps=500 \
wandb.project=depthsplat \
output_dir=checkpoints/dl3dv-256x448-depthsplat-base


# evaluate on dl3dv, view 6
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
data_loader.train.batch_size=1 \
dataset.test_chunk_interval=1 \
trainer.val_check_interval=0.2 \
train.eval_model_every_n_val=3 \
dataset.roots=[datasets/dl3dv] \
mode=test \
dataset.near=1. \
dataset.far=200. \
dataset/view_sampler=evaluation \
test.compute_scores=true \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v_video_0-50.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_regressor_channels=32 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=128 \
model.encoder.multiview_trans_nearest_n_views=3 \
model.encoder.costvolume_nearest_n_views=3 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-75cc0183.pth \
wandb.project=depthsplat \
wandb.mode=disabled \
test.save_image=false \
test.save_depth=false \
test.save_video=false \
test.stablize_camera=false \
output_dir=output/tmp


# evaluate on dl3dv, view 4
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
data_loader.train.batch_size=1 \
dataset.test_chunk_interval=1 \
trainer.val_check_interval=0.2 \
train.eval_model_every_n_val=3 \
dataset.roots=[datasets/dl3dv] \
mode=test \
dataset.near=1. \
dataset.far=200. \
dataset/view_sampler=evaluation \
test.compute_scores=true \
dataset.view_sampler.num_context_views=4 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_4v_tgt_4v_video_0-50.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_regressor_channels=32 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=128 \
model.encoder.multiview_trans_nearest_n_views=3 \
model.encoder.costvolume_nearest_n_views=3 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-75cc0183.pth \
wandb.project=depthsplat \
wandb.mode=disabled \
test.save_image=false \
test.save_depth=false \
test.save_video=false \
test.stablize_camera=false \
output_dir=output/tmp


# evaluate on dl3dv, view 2
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
data_loader.train.batch_size=1 \
dataset.test_chunk_interval=1 \
trainer.val_check_interval=0.2 \
train.eval_model_every_n_val=3 \
dataset.roots=[datasets/dl3dv] \
mode=test \
dataset.near=1. \
dataset.far=200. \
dataset/view_sampler=evaluation \
test.compute_scores=true \
dataset.view_sampler.num_context_views=2 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_2v_tgt_4v_video_0-50.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_regressor_channels=32 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=128 \
model.encoder.multiview_trans_nearest_n_views=3 \
model.encoder.costvolume_nearest_n_views=3 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-75cc0183.pth \
wandb.project=depthsplat \
wandb.mode=disabled \
test.save_image=false \
test.save_depth=false \
test.save_video=false \
test.stablize_camera=false \
output_dir=output/tmp


# render video on dl3dv (need to have ffmpeg installed)
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
data_loader.train.batch_size=1 \
dataset.test_chunk_interval=1 \
trainer.val_check_interval=0.2 \
train.eval_model_every_n_val=3 \
dataset.roots=[datasets/dl3dv_full] \
mode=test \
dataset.near=1. \
dataset.far=200. \
dataset/view_sampler=evaluation \
test.compute_scores=false \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v_video_0-50.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
model.encoder.gaussian_regressor_channels=32 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=128 \
model.encoder.multiview_trans_nearest_n_views=3 \
model.encoder.costvolume_nearest_n_views=3 \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-75cc0183.pth \
wandb.project=depthsplat \
wandb.mode=disabled \
test.save_image=false \
test.save_depth=false \
test.save_video=true \
test.stablize_camera=true \
output_dir=output/tmp





