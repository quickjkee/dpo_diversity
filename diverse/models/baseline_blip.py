from models.src.config.options import *
from models.src.config.utils import *
from models.src.models.blip_pretrain import blip_pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from transformers import BertTokenizer
from tqdm import tqdm
import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform():
    return Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


model = blip_pretrain(pretrained=config['blip_path'], image_size=config['BLIP']['image_size'],
                      vit=config['BLIP']['vit']).to('cuda')
preprocess = _transform()
tokenizer = init_tokenizer()

class BlipBase:

    def __init__(self):
        self.model = model

    @torch.no_grad()
    def _predict(self, item, factor):
        text_ids, text_mask, img_1, img_2 = item['text_ids'], item['text_mask'], \
                                            item['image_1'], item['image_2']
        text_ids = text_ids.view(text_ids.shape[0], -1).to('cuda')  # [batch_size, seq_len]
        text_mask = text_mask.view(text_mask.shape[0], -1).to('cuda')  # [batch_size, seq_len]
        img_1 = img_1.to('cuda')  # [batch_size, C, H, W]
        img_2 = img_2.to('cuda')  # [batch_size, C, H, W]

        # img1
        image_embeds_1 = self.model.visual_encoder(img_1.unsqueeze(0))
        image_atts_1 = torch.ones(image_embeds_1.size()[:-1], dtype=torch.long).to('cuda')
        emb_1 = self.model.text_encoder(text_ids,
                                        attention_mask=text_mask,
                                        encoder_hidden_states=image_embeds_1,
                                        encoder_attention_mask=image_atts_1,
                                        return_dict=True,
                                        ).last_hidden_state  # [batch_size, seq_len, feature_dim]
        emb_1 = emb_1[:, 0, :].float()

        # img2
        image_embeds_2 = self.model.visual_encoder(img_2.unsqueeze(0))
        image_atts_2 = torch.ones(image_embeds_2.size()[:-1], dtype=torch.long).to('cuda')
        emb_2 = self.model.text_encoder(text_ids,
                                        attention_mask=text_mask,
                                        encoder_hidden_states=image_embeds_2,
                                        encoder_attention_mask=image_atts_2,
                                        return_dict=True,
                                        ).last_hidden_state  # [batch_size, seq_len, feature_dim]
        emb_2 = emb_2[:, 0, :].float()

        # similarity
        pred = (F.cosine_similarity(emb_1, emb_2, dim=1).item() + 1) / 2
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

