import torch


def calc_pick_and_clip_scores(processor, model, images, prompts, batch_size=64, device='cuda'):
    assert len(prompts) == len(images)

    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )['pixel_values'].to(device)

    text_inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )['input_ids'].to(device)

    scores = torch.zeros(len(prompts))
    for i in range(0, len(prompts), batch_size):
        image_batch = image_inputs[i:i+batch_size]
        text_batch = text_inputs[i:i+batch_size]
        with torch.no_grad():
            # embed
            image_embs = model.get_image_features(image_batch)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = model.get_text_features(text_batch)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores[i:i+batch_size] = (text_embs * image_embs).sum(-1) #model.logit_scale.exp() * 
    return scores.cpu()
    

@torch.no_grad()
def calc_diversity_scores(processor, model, images, batch_size=64, num_seeds=5, device='cuda'):
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )['pixel_values'].to(device)
    
    dim = model.visual_projection.weight.shape[0]
    image_embs = torch.zeros(len(images), dim)
    for i in range(0, len(images), batch_size):
        image_batch = image_inputs[i:i+batch_size]
        embs = model.get_image_features(image_batch)
        embs = embs / torch.norm(embs, dim=-1, keepdim=True)
        image_embs[i:i+batch_size] = embs

    image_embs = image_embs.reshape(-1, num_seeds, dim)
    sim_matricies = 1 - torch.bmm(image_embs, image_embs.transpose(2, 1))
    # sim_matricies.diagonal(dim1=1, dim2=2).zero_()
    scores = sim_matricies.sum(-1).sum(-1) / num_seeds / (num_seeds - 1)
    return scores


