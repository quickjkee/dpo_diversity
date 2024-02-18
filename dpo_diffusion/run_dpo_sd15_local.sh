export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# "/extra_disk_1/dbaranchuk/dpms/pretrained/sd-v1-5"
ACCELERATE_CONFIG="configs/default_config.yaml"
PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT

accelerate launch --config_file $ACCELERATE_CONFIG --num_processes=6 --main_process_port $PORT train_diffusion_dpo_local.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir="results_sd15" \
  --mixed_precision="fp16" \
  --dataset_name="yuvalkirstain/pickapic_v2" \
  --resolution=512 \
  --train_batch_size=64 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --rank=8 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --checkpointing_steps=2000 \
  --run_validation \
  --validation_steps=50 \
  --seed="0" \
  --val_seed="0,1,2,3,4,5,6,7,8,9" \
  --max_train_samples 20000 \
  --cache_dir "/extra_disk_1/dbaranchuk/datasets/pickapic_v2/"
