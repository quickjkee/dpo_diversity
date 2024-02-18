import io
import os
import torch
import accelerate
from accelerate.logging import get_logger
import src.dist_util as dist
from transformers import AutoProcessor, AutoModel

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
try:
    import plotly.graph_objects as go
except:
    print('failed to import plotly')

from src.scores import calc_pick_and_clip_scores, calc_diversity_scores


logger = get_logger(__name__)


def to_image(tensor, adaptive=False):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    else:
        tensor = ((tensor + 1) / 2).clamp(0, 1)

    return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


def to_image_grid(tensor, adaptive=False, **kwargs):
    return to_image(make_grid(tensor, **kwargs), adaptive)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_log_validation_prompts(path, num_prompts=10):
    df = pd.read_csv(path)
    all_text = list(df['caption'])
    return all_text[:num_prompts]


def get_validation_prompts(path, bs=20, num_prompts=500):
    df = pd.read_csv(path)
    all_text = list(df['caption'])
    all_text = all_text[:num_prompts]

    num_batches = ((len(all_text) - 1) // (bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()] 

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]
    return rank_batches, rank_batches_index


def sample(args, unet, accelerator, weight_dtype, step, num_prompts=500, num_seeds=5):
    all_prompts = list(pd.read_csv(args.validation_prompts)['caption'])[:num_prompts]
    rank_batches, rank_batches_index = get_validation_prompts(
        args.validation_prompts, num_prompts=num_prompts
    )
    
    logger.info(f"Generating {len(all_prompts)} validation prompts for {num_seeds} seeds...")

    # create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.unet = accelerator.unwrap_model(unet)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
   
    local_images = []
    local_text_idxs = []
    for seed_id in range(num_seeds):
        seed = args.seed + seed_id if args.seed else None
        generator = torch.Generator(device=accelerator.device).manual_seed(seed) if seed else None
        for cnt, mini_batch in enumerate(tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
            with torch.cuda.amp.autocast():
                images = pipeline(list(mini_batch), num_inference_steps=25, generator=generator).images

            for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
                img_tensor = torch.tensor(np.array(images[text_idx]))
                local_images.append(img_tensor)
                local_text_idxs.append(global_idx)

    local_images = torch.stack(local_images).to(dist.dev())
    hwc = local_images.shape[1:]
    local_images = local_images.reshape(num_seeds, -1, *hwc).transpose(1, 0).reshape(-1, *hwc)

    local_text_idxs = torch.tensor(local_text_idxs).to(dist.dev())
    local_text_idxs = local_text_idxs.reshape(num_seeds, -1).T.reshape(-1)

    gathered_images = [torch.zeros_like(local_images) for _ in range(dist.get_world_size())]
    gathered_text_idxs = [torch.zeros_like(local_text_idxs) for _ in range(dist.get_world_size())]
        
    dist.all_gather(gathered_images, local_images)  # gather not supported with NCCL
    dist.all_gather(gathered_text_idxs, local_text_idxs) 
            
    if dist.get_rank() == 0:
        gathered_images = np.concatenate(
            [images.cpu().numpy() for images in gathered_images], axis=0
        )
        gathered_images = [ToPILImage()(image) for image in gathered_images] 
        gathered_text_idxs =  np.concatenate(
            [text_idxs.cpu().numpy() for text_idxs in gathered_text_idxs], axis=0
        )
        gathered_prompts = [all_prompts[idx] for idx in gathered_text_idxs]
        # os.makedirs("samples", exist_ok=True)
        # for i in range(len(all_prompts)):
        #     for seed_id in range(num_seeds):
        #         gathered_images[i * num_seeds + seed_id].save(f"samples/{i}_seed_{seed_id}.jpg")
    else:
        gathered_images = None
        gathered_prompts = None

    # Done.
    dist.barrier()
    return gathered_images, gathered_prompts
    

def evaluate(args, accelerator, images, prompts, device='cuda', num_seeds=5):
    processor = AutoProcessor.from_pretrained(args.clip_model_name_or_path)
    clip_model = AutoModel.from_pretrained(args.clip_model_name_or_path).eval().to(device)
    pickscore_model = AutoModel.from_pretrained(args.pickscore_model_name_or_path).eval().to(device)
    
    pick_scores = calc_pick_and_clip_scores(processor, pickscore_model, images, prompts, device=device)
    clip_scores = calc_pick_and_clip_scores(processor, clip_model, images, prompts, device=device)
    diversity_scores = calc_diversity_scores(processor, clip_model, images, num_seeds=num_seeds, device=device)
    return pick_scores.mean(), clip_scores.mean(), diversity_scores.mean()
