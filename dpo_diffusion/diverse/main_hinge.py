import sys

sys.path.append('/home/quickjkee/diversity/models/src')
sys.path.append(f'/home/quickjkee/diversity/models/src/config')
sys.path.append(f'/home/quickjkee/diversity/models')

from models.src.config.options import *
from models.src.config.utils import *
from utilss.lora_wrapper import ViTConfig, ViTModel
from peft import LoraConfig, get_peft_model

from sklearn.model_selection import train_test_split

from utilss.parser import Parser
from dataset import DiversityDataset
from models.src.HingeReward import HingeReward
from train_hinge import run_train

#from models.baseline_blip import blip_pretrain, preprocess
from models.baseline_clip import preprocess, model

def get_weights(dataset, label):
    all_labels = []
    for item in dataset:
        all_labels.append(item[label])

    labels = set(all_labels)
    class_weights = {}
    # Prepare weights for each class
    for label in labels:
        class_weights[label] = 0
        for el in all_labels:
            if el == label:
                class_weights[label] += 1
        class_weights[label] = (len(all_labels)) / class_weights[label]

    loss_weights = list(class_weights.values())
    return loss_weights


# Prepare dataset
# -----------------------------------
LABEL = 'angle'
parser = Parser()
paths = ['files/0_500_pickscore_coco',
         'files/diverse_coco_pick_3_per_prompt_500_1000.out',
         'files/diverse_coco_pick_3_per_prompt_1000_1500',
         'files/diverse_coco_pick_3_per_prompt_1500_2000',
         'files/diverse_coco_pick_3_per_prompt_2000_2500',
         'files/diverse_coco_pick_3_per_prompt_2500_3000',
         'files/diverse_coco_pick_3_per_prompt_3000_3500', ]
df = parser.raw_to_df(paths, do_overlap=True, keep_no_info=False)[:250]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
train_dataset = DiversityDataset(train_df,
                                 local_path='/extra_disk_1/quickjkee/diversity_images',
                                 preprocess=preprocess, is_train=True)
loss_w = get_weights(train_dataset, LABEL)
loss_w = torch.FloatTensor(loss_w)
valid_dataset = DiversityDataset(test_df,
                                 local_path='/extra_disk_1/quickjkee/diversity_images',
                                 preprocess=preprocess, is_train=False)
# -----------------------------------


# Prepare models
# -----------------------------------
use_lora = False

# FOR BLIP
#backbone = blip_pretrain(pretrained=config['blip_path'], image_size=config['BLIP']['image_size'],
#                         vit=config['BLIP']['vit'])
#model = backbone.visual_encoder
#if use_lora:
#    lora_config = {
#        "r": 16,
#        "lora_alpha": 0.5,
#        "lora_dropout": 0.3,
#        "bias": "none",
#        "target_modules": ['qkv']
#    }
#    lora_config = LoraConfig(**lora_config)
#    model = get_peft_model(ViTModel(model, ViTConfig()), lora_config)

# FOR DREAMSIMSIM
model = model.extractor_list[0].model

backbone = model
backbone.requires_grad_(True)

main_model = HingeReward(backbone, img_lora=use_lora)
load_model(main_model, 'final_checkpoints/angle_74/best_lr\=0.0001.pt')
# -----------------------------------


# Run
print(f'Label for training {LABEL}')
run_train(train_dataset=train_dataset,
          valid_dataset=valid_dataset,
          model=main_model,
          loss_w=loss_w,
          label=LABEL)
