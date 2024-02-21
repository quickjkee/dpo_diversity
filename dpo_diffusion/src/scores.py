import torch
from PIL import Image
import io
from tqdm import tqdm


def calc_pick_and_clip_scores(model, image_inputs, text_inputs, batch_size=64):
    assert len(image_inputs) == len(text_inputs)

    scores = torch.zeros(len(text_inputs))
    for i in range(0, len(text_inputs), batch_size):
        image_batch = image_inputs[i:i + batch_size]
        text_batch = text_inputs[i:i + batch_size]
        with torch.no_grad():
            # embed
            image_embs = model.get_image_features(image_batch)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = model.get_text_features(text_batch)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            # score
            scores[i:i + batch_size] = (text_embs * image_embs).sum(-1)  # model.logit_scale.exp() *
    return scores.cpu()


@torch.no_grad()
def calc_clip_diversity_scores(model, image_inputs, batch_size=64, num_seeds=5):
    assert num_seeds > 1
    dim = model.visual_projection.weight.shape[0]
    image_embs = torch.zeros(len(image_inputs), dim)
    for i in range(0, len(image_inputs), batch_size):
        image_batch = image_inputs[i:i + batch_size]
        embs = model.get_image_features(image_batch)
        embs = embs / torch.norm(embs, dim=-1, keepdim=True)
        image_embs[i:i + batch_size] = embs

    image_embs = image_embs.reshape(-1, num_seeds, dim)
    sim_matricies = 1 - torch.bmm(image_embs, image_embs.transpose(2, 1))
    # sim_matricies.diagonal(dim1=1, dim2=2).zero_()
    scores = sim_matricies.sum(-1).sum(-1) / num_seeds / (num_seeds - 1)
    return scores


def calc_probs(prompt, images, processor, accelerator, model):
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(accelerator.device)

    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(accelerator.device)

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = (text_embs @ image_embs.T)[0]

    return scores.cpu().item()


@torch.no_grad()
def calc_diversity_scores(images, model, processor, batch_size=64, num_seeds=5, device='cuda'):
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )['pixel_values'].to(device)

    image_embs = []
    for i in range(0, len(image_inputs), batch_size):
        image_batch = image_inputs[i:i + batch_size]
        embs = model.get_image_features(image_batch)
        embs = embs / torch.norm(embs, dim=-1, keepdim=True)
        image_embs.append(embs)

    dim = embs.size(dim=-1)
    image_embs = torch.cat(image_embs, dim=0)
    image_embs = image_embs.reshape(-1, num_seeds, dim)
    sim_matricies = 1 - torch.bmm(image_embs, image_embs.transpose(2, 1))

    scores = sim_matricies.sum(-1).sum(-1) / num_seeds / (num_seeds - 1)
    return scores


def collect_pickscores(train_dataset, model_pick, processor_pick, quality_threshold_for_div):
    all_scores = []
    for batch in tqdm(train_dataset):
        all_pixel_values = []
        for col_name in ["jpg_0", "jpg_1"]:
            images = [Image.open(io.BytesIO(im_bytes)).convert("RGB") for im_bytes in batch[col_name]]
            pixel_values = processor_pick(
                images=images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )['pixel_values'].to(model_pick.device)
            all_pixel_values.append(pixel_values)

        prompt = batch['caption']
        text_inputs = processor_pick(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(model_pick.device)
        all_pixel_values = torch.cat(all_pixel_values)

        image_embs = model_pick.get_image_features(all_pixel_values)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = model_pick.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = (text_embs @ image_embs.T)[0].cpu()
        all_scores.append(scores)


