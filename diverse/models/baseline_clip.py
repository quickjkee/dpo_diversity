import open_clip
import torch
import torch.nn.functional as F
from dreamsim import dreamsim
from tqdm import tqdm

model, preprocess = dreamsim(pretrained=True, dreamsim_type="open_clip_vitb32")
#model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')
model.to('cuda')

class ClipBase:

    def __init__(self):
        self.model = model

    @torch.no_grad()
    def _predict(self, item, factor):
        # img1
        image_1 = item['image_1'].to('cuda')
        image_features_1 = model.encode_image(image_1.unsqueeze(0))
        image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)

        # img2
        image_2 = item['image_2'].to('cuda')
        image_features_2 = model.encode_image(image_2.unsqueeze(0))
        image_features_2 /= image_features_1.norm(dim=-1, keepdim=True)

        # similarity
        pred = (F.cosine_similarity(image_features_1, image_features_2, dim=1).item() + 1) / 2
        true = item[factor]
        return pred, true

    @torch.no_grad()
    def __call__(self, dataset, factor):
        preds = []
        trues = []
        for batch in tqdm(dataset):
            pred, true = self._predict(batch, factor)
            preds.append(pred)
            trues.append(true)

        return preds, trues
