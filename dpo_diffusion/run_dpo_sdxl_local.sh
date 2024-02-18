export CUDA_VISIBLE_DEVICES=3
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# "/extra_disk_1/dbaranchuk/dpms/pretrained/sd-v1-5"
ACCELERATE_CONFIG="configs/default_config.yaml"
PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT

accelerate launch --config_file $ACCELERATE_CONFIG --num_processes=1 --main_process_port $PORT train_diffusion_dpo_sdxl_local.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --output_dir="results_sdxl" \
  --mixed_precision="fp16" \
  --dataset_name="yuvalkirstain/pickapic_v2" \
  --resolution=1024 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --rank=8 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --checkpointing_steps=2000 \
  --run_validation --validation_steps=200 \
  --seed="0" \
  --max_train_samples 20000 \
  --cache_dir "/extra_disk_1/dbaranchuk/datasets/pickapic_v2/"