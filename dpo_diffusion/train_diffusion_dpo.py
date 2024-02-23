#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 bram-w, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import contextlib
import io
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available

import yaml
import pickle
from omegaconf import OmegaConf
import torch.distributed as dist
from src.nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot, copy_logs_to_logs_path
from src import dnnlib
import src
from yt_tools.utils import instantiate_from_config
from src.utils import image_grid, get_log_validation_prompts, get_validation_prompts
from transformers import AutoProcessor, AutoModel
from diverse.models.src.HingeReward import HingeReward
from src.scores import calc_probs, calc_diversity_scores, collect_pickscores
import torchvision

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__)

VALIDATION_PROMPTS = [
    "A picture of a cute girl at a meadow, smiling, at night, in anime style",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    'a hedgehog gummy',
    'a car that is made out of wood',
    'A girl with pale blue hair and a cami tank top',
    'A bird with 8 spider legs',
    'An underwater city with fish swimming around',
    'A futuristic motorcycle prototype, design inspired by F-117 Nighthawk,',
    'Ronald Reagan as a LEGO minifigure',
    'a man with a dog watching a beautiful sunset, lush green landscape',
    'cat as a mafia boss',
    'zombie in school at night with a flamethrower',
    'A human hand with five fingers',
    'Watercolor painting of european modern city, medieval, nightfall moonlight, by greg rutkowski, by anders zorn',
    'A cat dressed as a ninja',
    'A photo of a viking man holding a broadsword, snowy forest'
]

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def log_validation(args, unet, accelerator, weight_dtype, epoch, is_final_validation=False):
    logger.info(f"Running validation... \n Generating images with prompts:\n" f" {VALIDATION_PROMPTS}.")

    # PICKSCORE
    ############
    processor_pick = AutoProcessor.from_pretrained(args.pickscore_model_name_or_path)
    model_pick = AutoModel.from_pretrained(args.pickscore_model_name_or_path).eval().to(accelerator.device)
    ##########

    # OURS
    ############
    with dnnlib.util.open_url(args.dreamsim_open_clip_vitb32_path) as f:
        model_ours = pickle.load(f)['model'].to(accelerator.device)
    model_ours = HingeReward(model_ours, threshold=0.981654167175293, img_lora=False).eval().to(accelerator.device)
    ############

    # create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype, safety_checker=None
    )
    if not is_final_validation:
        pipeline.unet = accelerator.unwrap_model(unet)
    else:
        pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    images = []
    raw_images = []
    scores = []
    context = contextlib.nullcontext() if is_final_validation else torch.cuda.amp.autocast()

    for prompt in VALIDATION_PROMPTS:
        with context:
            list_seed = args.val_seed.split(',')
            for seed in list_seed:
                seed = int(seed)
                generator = torch.Generator(device=accelerator.device).manual_seed(seed)
                image = pipeline(prompt, num_inference_steps=25, generator=generator).images[0]
                raw_images.append(image)
                score = calc_probs(prompt, image, processor_pick, accelerator, model_pick)
                images.append(torch.tensor(np.array(image)).unsqueeze(0).permute(0, 3, 1, 2))
                scores.append(score)

    diversity_score = calc_diversity_scores(raw_images, model_ours, processor_pick, 1, len(list_seed), accelerator.device)

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            torch_images = torch.cat(images, dim=0)
            grid = torchvision.utils.make_grid(torch_images, nrow=len(list_seed))
            grid = grid.permute(1, 2, 0)
            tracker.writer.add_image(tracker_key, grid, epoch, dataformats="HWC")

    logs = {'val_pickscore': np.mean(scores),
            'diversity_score': diversity_score.mean().item()}
    accelerator.log(logs, step=epoch)

    # Also log images without the LoRA params for comparison.
    if is_final_validation:
        pipeline.disable_lora()
        no_lora_images = [
            pipeline(prompt, num_inference_steps=25, generator=generator).images[0] for prompt in VALIDATION_PROMPTS
        ]

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in no_lora_images])
                tracker.writer.add_images("test_without_lora", np_images, epoch, dataformats="NHWC")
            

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pickscore_model_name_or_path",
        type=str,
        default="yuvalkirstain/PickScore_v1",
        required=True,
    )
    parser.add_argument(
        "--clip_model_name_or_path",
        type=str,
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        required=True,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/picapic.yaml",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--run_validation",
        default=False,
        action="store_true",
        help="Whether to run validation inference in between training and also after training. Helps to track progress.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diffusion-dpo-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--val_seed", type=str, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=8,
        help="Batch size to use for VAE encoding of the images for efficient processing.",
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--dreamsim_open_clip_vitb32_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--beta_dpo",
        type=int,
        default=2500,
        help="DPO KL Divergence penalty.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        help="DPO loss type. Can be one of 'sigmoid' (default), 'ipo', or 'cpo'",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="diffusion-dpo-lora",
        help=("The name of the tracker to report results to."),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default="prompts/validation.csv",
    )
    parser.add_argument(
        "--num_eval_seeds",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--quality_threshold_for_div",
        type=float,
        default=0.0
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None:
        raise ValueError("Must provide a `dataset_name`.")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def tokenize_captions(tokenizer, examples):
    max_length = tokenizer.model_max_length
    captions = []
    for caption in examples["caption"]:
        captions.append(caption.decode("utf-8"))

    text_inputs = tokenizer(
        captions, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    return text_inputs.input_ids


@torch.no_grad()
def encode_prompt(text_encoder, input_ids):
    text_input_ids = input_ids.to(text_encoder.device)
    attention_mask = None

    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def collate_fn(examples):
    pixel_values = torch.stack(examples["pixel_values"])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    final_dict = {"pixel_values": pixel_values}

    pixel_values_ours = torch.stack(examples["pixel_values_ours"])
    pixel_values_ours = pixel_values_ours.to(memory_format=torch.contiguous_format).float()
    final_dict["pixel_values_ours"] = pixel_values_ours

    final_dict["input_ids"] = examples["input_ids"]

    if args.quality_threshold_for_div:
        final_dict["mask"] = examples["mask"]

    return final_dict


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    logger.info(f"Logging dir: {logging_dir}")
    if accelerator.is_main_process:
        copy_snapshot_to_out(args.output_dir)
    dist.barrier()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # OURS MODELS
    ######################
    processor_ours = AutoProcessor.from_pretrained(args.clip_model_name_or_path)
    with dnnlib.util.open_url(args.dreamsim_open_clip_vitb32_path) as f:
        model_ours = pickle.load(f)['model'].to(accelerator.device)
    model_ours = HingeReward(model_ours, threshold=0.981654167175293, img_lora=False).eval().to(accelerator.device).to(torch.float16)
    ######################

    # PICKSCORE
    ######################
    model_pick = AutoModel.from_pretrained(args.pickscore_model_name_or_path).eval().to(accelerator.device)
    ######################

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Set up LoRA.
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=None,
            )
            
    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    current_step = 0 #recover_resume_step()
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"Resume the training from {total_batch_size * current_step}")

    with open(args.dataset_config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config = OmegaConf.create(config)
        config['train_dataloader'][0]['params']['batch_size'] = args.train_batch_size
    
    if accelerator.is_main_process:
        print("Data config: ", config)
    train_dataset = train_dataloader = instantiate_from_config(config['train_dataloader'][0],
        skip_rows=total_batch_size * current_step
    )

    train_transforms = transforms.Compose(
        [
            transforms.Resize(int(args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if args.quality_threshold_for_div:
        q_value = args.quality_threshold_for_div

    @torch.no_grad()
    def preprocess_train(examples):
        all_pixel_values = []
        all_pixel_values_ours = []
        for col_name in ["jpg_0", "jpg_1"]:
            images = [Image.open(io.BytesIO(im_bytes)).convert("RGB") for im_bytes in examples[col_name]]
            pixel_values = [train_transforms(image) for image in images]
            pixel_values_ours = processor_ours(
                images=images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )['pixel_values']
            all_pixel_values.append(pixel_values)
            all_pixel_values_ours.append(pixel_values_ours)

        # Double on channel dim, jpg_y then jpg_w
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
            if label_0 == 0:
                im_tup = im_tup[::-1]
            combined_im = torch.cat(im_tup, dim=0)  # no batch dim
            combined_pixel_values.append(combined_im)
        examples["pixel_values"] = combined_pixel_values

        im_tup_iterator = zip(*all_pixel_values_ours)
        combined_pixel_values_ours = []
        for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
            if label_0 == 0:
                im_tup = im_tup[::-1]
            combined_im = torch.cat(im_tup, dim=0)  # no batch dim
            combined_pixel_values_ours.append(combined_im)
        examples["pixel_values_ours"] = combined_pixel_values_ours

        examples["input_ids"] = tokenize_captions(tokenizer, examples)

        # Make mask
        if args.quality_threshold_for_div:
            pixel_values = torch.stack(examples["pixel_values_ours"])
            feed_pixel_values = torch.cat(pixel_values.chunk(2, dim=1)).to(dtype=weight_dtype)
            img_1_batch, img_2_batch = feed_pixel_values.chunk(2, dim=0)
            txt_batch = examples["input_ids"]

            text_embs = model_pick.get_text_features(txt_batch.to(accelerator.device))
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            image_1_embs = model_pick.get_image_features(img_1_batch.to(accelerator.device))
            image_1_embs = image_1_embs / torch.norm(image_1_embs, dim=-1, keepdim=True)
            score_1 = (text_embs * image_1_embs).sum(-1)
            mask_1 = (score_1 > q_value) * 1.0

            image_2_embs = model_pick.get_image_features(img_2_batch.to(accelerator.device))
            image_2_embs = image_2_embs / torch.norm(image_2_embs, dim=-1, keepdim=True)
            score_2 = (text_embs * image_2_embs).sum(-1)
            mask_2 = (score_2 > q_value) * 1.0

            mask = mask_1 * mask_2
            examples['mask'] = mask

        return examples
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info(f"Num upd steps in epoch {num_update_steps_per_epoch} | Max train steps {args.max_train_steps} | Num epochs {args.num_train_epochs}")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_name, config=vars(args))

    # Prepare validation prompts 
    validation_prompts = get_log_validation_prompts(args.validation_prompts)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    for param in unet.parameters():
        # only upcast trainable parameters (LoRA) into fp32
        if param.requires_grad:
            param.data = param.to(torch.float32)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, examples in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                batch = preprocess_train(examples)
                batch = collate_fn(batch)
                batch["pixel_values"] = batch["pixel_values"].cuda()
                batch["input_ids"] = batch["input_ids"].cuda()

                # (batch_size, 2*channels, h, w) -> (2*batch_size, channels, h, w)
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                feed_pixel_values = torch.cat(pixel_values.chunk(2, dim=1))

                latents = []
                for i in range(0, feed_pixel_values.shape[0], args.vae_encode_batch_size):
                    latents.append(
                        vae.encode(feed_pixel_values[i : i + args.vae_encode_batch_size]).latent_dist.sample()
                    )
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1, 1)

                # Sample a random timestep for each image
                bsz = latents.shape[0] // 2
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long
                ).repeat(2)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = encode_prompt(text_encoder, batch["input_ids"]).repeat(2, 1, 1)

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Compute losses.
                model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
                model_losses_w, model_losses_l = model_losses.chunk(2)

                # For logging
                raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                model_diff = model_losses_w - model_losses_l  # These are both LBS (as is t)


                # Reference model predictions.
                accelerator.unwrap_model(unet).disable_adapters()
                with torch.no_grad():
                    ref_preds = unet(
                        noisy_model_input,
                        timesteps,
                        encoder_hidden_states,
                    ).sample.detach()
                    ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction="none")
                    ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

                    ref_losses_w, ref_losses_l = ref_loss.chunk(2)
                    ref_diff = ref_losses_w - ref_losses_l
                    raw_ref_loss = ref_loss.mean()

                # Re-enable adapters.
                accelerator.unwrap_model(unet).enable_adapters()

                # DIVERSITY LOSS
                ############################
                with torch.no_grad():
                    # Get data
                    batch["pixel_values_ours"] = batch["pixel_values_ours"].cuda()
                    pixel_values_ours = batch["pixel_values_ours"].to(dtype=weight_dtype)
                    feed_pixel_values_ours = torch.cat(pixel_values_ours.chunk(2, dim=1))
                    img_1_batch, img_2_batch = feed_pixel_values_ours.chunk(2, dim=0)

                    # Preds
                    labels = model_ours.classify(img_1_batch, img_2_batch)
                    print(labels.sum() / len(labels))

                    scores = -1 * (model_losses_w + model_losses_l - ref_losses_w - ref_losses_l)
                    positive_log = F.logsigmoid(args.beta_dpo * scores) * labels
                    negative_log = F.logsigmoid(-1 * args.beta_dpo * scores) * (1 - labels)
                    total_log = positive_log + negative_log

                    if args.quality_threshold_for_div:
                        mask = batch["mask"].cuda()
                        print(mask)
                        loss_div = -1 * (total_log * mask).mean()   #/ (mask.sum() + 0.00001)  # to avoid zero derivation
                    else:
                        loss_div = -1 * total_log.mean()
                ############################

                # Final loss.
                logits = ref_diff - model_diff
                if args.loss_type == "sigmoid":
                    loss_old = -1 * F.logsigmoid(args.beta_dpo * logits).mean()
                    loss = loss_old #+ loss_div
                elif args.loss_type == "hinge":
                    loss = torch.relu(1 - args.beta_dpo * logits).mean()
                elif args.loss_type == "ipo":
                    losses = (logits - 1 / (2 * args.beta)) ** 2
                    loss = losses.mean()
                else:
                    raise ValueError(f"Unknown loss type {args.loss_type}")

                implicit_acc = (logits > 0).sum().float() / logits.size(0)
                implicit_acc += 0.5 * (logits == 0).sum().float() / logits.size(0)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        # Nirvana utils
                        copy_out_to_snapshot(args.output_dir)
                        copy_logs_to_logs_path(logging_dir)

                    if args.run_validation and global_step % args.validation_steps == 0:
                        log_validation(
                            args, unet=unet, accelerator=accelerator, weight_dtype=weight_dtype, epoch=global_step
                        )
                        copy_out_to_snapshot(args.output_dir)

                dist.barrier()
                progress_bar.update(1)
                global_step += 1

                """
                if args.run_validation and global_step % args.validation_steps == 0:
                    images, prompts = sample(
                        args, unet=unet, accelerator=accelerator, 
                        weight_dtype=weight_dtype, step=global_step, 
                        num_seeds=args.num_eval_seeds
                    )
                    if accelerator.is_main_process:
                        pickscore, clip_score, diversity_score = evaluate(
                            args, accelerator, images, prompts, num_seeds=args.num_eval_seeds
                        )
                        val_logs = {
                            "pickscore": pickscore.item(),
                            "clip_score": clip_score.item(),
                            "diversity_score": diversity_score.item()
                        }
                        accelerator.log(val_logs, step=global_step)

                        copy_out_to_snapshot(args.output_dir)
                """
                    
            logs = {
                "loss": loss.detach().item(),
                "loss_pick": loss_old.detach().item(),
                "loss_div": loss_div.detach().item(),
                "raw_model_loss": raw_model_loss.detach().item(),
                "ref_loss": raw_ref_loss.detach().item(),
                "implicit_acc": implicit_acc.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir, unet_lora_layers=unet_lora_state_dict, text_encoder_lora_layers=None
        )

        # Final validation?
        if args.run_validation:
            log_validation(
                validation_prompts,
                args,
                unet=None,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
