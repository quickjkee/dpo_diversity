import matplotlib.pyplot as plt
import numpy as np

from utilss.parser import Parser

parser = Parser()
paths = ['../files/0_500_pickscore_coco',
         '../files/diverse_coco_pick_3_per_prompt_500_1000.out',
         '../files/diverse_coco_pick_3_per_prompt_1000_1500',
         '../files/diverse_coco_pick_3_per_prompt_1500_2000',
         '../files/diverse_coco_pick_3_per_prompt_2000_2500']
sbs = parser.raw_to_df(paths, do_overlap=True, keep_no_info=False)

models = ['addxl', 'lcmxl', 'sd21', 'sdxl', 'all']

for model in models:
    fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)
    for i, factor in enumerate(['angle', 'style', 'similar', 'background', 'main_object']):
        dict_ = {}
        values = np.array(sbs[factor])
        names = np.array(sbs['image_1'])
        if model == 'all':
            new_values = values
        else:
            new_values = [value for i, value in enumerate(values) if model in names[i]]
        for value in new_values:
            try:
                dict_[value] += 1
            except KeyError:
                dict_[value] = 0
                dict_[value] += 1
        axes[i].set_title(factor)
        axes[i].bar(dict_.keys(), dict_.values())

    plt.suptitle(f'{model}')
    plt.savefig(f'{model}.png')