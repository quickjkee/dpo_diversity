import torch

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
def calc_diversity_scores(image_inputs, model, batch_size=64, num_seeds=5, device='cuda'):
    image_embs = []
    for i in range(0, len(image_inputs), batch_size):
        image_batch = image_inputs[i:i+batch_size]
        embs = model.get_image_features(image_batch)
        embs = embs / torch.norm(embs, dim=-1, keepdim=True)
        image_embs.append(embs)

    dim = embs.size(dim=-1)
    image_embs = torch.cat(image_embs, dim=0)
    image_embs = image_embs.reshape(1, num_seeds, dim)
    sim_matricies = 1 - torch.bmm(image_embs, image_embs.transpose(2, 1))

    scores = sim_matricies.sum(-1).sum(-1) / num_seeds / (num_seeds - 1)
    return scores.item()


