export CUDA_VISIBLE_DEVICES=4,5,6,7

export PYTHONPATH=$PYTHONPATH:"/home/dbaranchuk/dpms/"
ACCELERATE_CONFIG="configs/default_config.yaml"
PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT

accelerate launch --config_file $ACCELERATE_CONFIG --num_processes=4 --main_process_port $PORT train_diffusion_dpo.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
  --dataset_name="pickapic_v2" \
  --output_dir="results_sd15" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --rank=8 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=4000 \
  --checkpointing_steps=1000 \
  --run_validation --validation_steps=100 \
  --seed="0" \
  --beta_dpo 2000 \
  --dataset_config configs/pickapic.yaml \
  --resume_from_checkpoint "latest" \
  --pickscore_model_name_or_path "/extra_disk_1/dbaranchuk/pretrained_models/PickScore_v1" \
  --clip_model_name_or_path "/extra_disk_1/dbaranchuk/pretrained_models/CLIP-ViT-H-14-laion2B-s32B-b79K"