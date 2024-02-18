import sys
sys.path.append('/home/quickjkee/diversity/models/src')
sys.path.append('/home/quickjkee/diversity')
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy import stats
from utilss.parser import Parser
from dataset import DiversityDataset
from metrics import samples_metric
from models.baseline_clip import preprocess, ClipBase
#from models.baseline_blip import preprocess, BlipBase

parser = Parser()
paths = ['files/0_500_pickscore_coco',
         'files/diverse_coco_pick_3_per_prompt_500_1000.out',
         'files/diverse_coco_pick_3_per_prompt_1000_1500',
         'files/diverse_coco_pick_3_per_prompt_1500_2000',
         'files/diverse_coco_pick_3_per_prompt_2000_2500', 
         'files/diverse_coco_pick_3_per_prompt_2500_3000', 
         'files/diverse_coco_pick_3_per_prompt_3000_3500', ]
df = parser.raw_to_df(paths, do_overlap=True, keep_no_info=False)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
dataset_train = DiversityDataset(train_df,
                                 local_path='/extra_disk_1/quickjkee/diversity_images',
                                 preprocess=preprocess)
dataset_test = DiversityDataset(test_df,
                                local_path='/extra_disk_1/quickjkee/diversity_images',
                                preprocess=preprocess)
clip_baseline = ClipBase()

factor = 'main_object'
pred, true = clip_baseline(dataset_train, factor)
pred = np.array(pred)
true = np.array(true)
idx = true != -1
true = list(true[idx])
pred = pred[idx]

accs = []
threshs = np.linspace(min(pred), max(pred), 10)
for thresh in threshs:
    curr_pred = (np.array(pred) > thresh) * 1
    curr_pred = list(curr_pred.astype(int))
    accs.append(samples_metric(true, curr_pred)[0])

stupid = [0] * len(true)
ac_stupid = samples_metric(true, stupid)[0]
max_value = max(accs)

found_thresh = threshs[accs.index(max_value)]
pred_val, true_val = clip_baseline(dataset_test, factor)
pred_val_th = (np.array(pred_val) > found_thresh) * 1
pred_val_th = list(pred_val_th.astype(int))
val_acc, val_std = samples_metric(true_val, pred_val_th)
print(f'acc {val_acc}, std {val_std}')
res = stats.spearmanr(pred_val, true_val)
print(f'corr {res.statistic}')

fig, axes = plt.subplots(1, 1,  figsize=(10, 5))
metrics = ['accuracy']
values = [accs]
stupid = [ac_stupid]
for j in range(1):
    axes.set_title(metrics[j])
    axes.plot(threshs, values[j])
    axes.axhline(stupid[j], color='r', linestyle='-', label='majority class')
    axes.axhline(val_acc, color='g', linestyle='-', label='validation')

plt.legend()
plt.suptitle(f'{factor}')
plt.savefig(f'{factor}.png')
