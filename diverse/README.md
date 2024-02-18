Parse the results:

```Python
from utilss.parser import Parser

parser = Parser()

# Path to raw annotation files
paths = ['../files/0_500_pickscore_coco',
         '../files/diverse_coco_pick_3_per_prompt_500_1000.out',
         '../files/diverse_coco_pick_3_per_prompt_1000_1500',
         '../files/diverse_coco_pick_3_per_prompt_1500_2000',
         '../files/diverse_coco_pick_3_per_prompt_2000_2500']

# Turn into dataframe, where each row it is a single vote
df = parser.raw_to_df(paths, do_overlap=True, keep_no_info=False)
# do_overlap - overlap over annotators or no
# keep_no_info - delete -1 or on
```

You can also check ```third_party/histograms.py``` for plots creating.
