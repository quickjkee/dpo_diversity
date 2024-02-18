import numpy as np
import json
import pandas as pd
from collections import Counter

class Parser:

    # -----------------------------------------------------
    def __init__(self,
                 factors=None,
                 keys=None):

        if keys is None:
            keys = {
                'no_info': -1,
                'same': 1,
                'different': 0
            }
            print(f'Parser initialized with the keys {keys}')
        if factors is None:
            factors = ['angle', 'style', 'similar', 'background', 'main_object']
            print(f'Parser initialized with the factors {factors}')

        self.keys = keys
        self.factors = factors
        # Cheating annotators
        cheat_workers = ['7b3c70c950afd356c52847a2665e85e6', '29464473a65afe4015ea060dac0aa6c6']
        print(f'Workers {cheat_workers} will be skipped')
        self.gays = cheat_workers
    # -----------------------------------------------------

    # -----------------------------------------------------
    def raw_to_df(self, paths, do_overlap=True, keep_no_info=True):
        def aggregate(inp):
            inp = list(inp)
            max_val = max(Counter(inp).values())
            if max_val == 1 and isinstance(inp[0], int):
                #diff_val = self.keys['same']
                #if diff_val == max_val:
                #    return diff_val
                #else:
                return -1
            else:
                return max(set(inp), key=inp.count)

        # Collect all raw annotations into pandas df
        df_raw = pd.DataFrame()
        for path in paths:
            with open(f"{path}", "r") as io_str:
                data = json.load(io_str)
                df_path = pd.DataFrame.from_dict(data)
                df_raw = pd.concat([df_raw, df_path])

        df_raw['idx'] = range(0, len(df_raw))
        df_raw = df_raw.set_index('idx')

        # To easy to use pandas
        total_dict = {}
        for i in range(len(df_raw)):
            name = df_raw["workerId"][i]
            if name in self.gays:
                continue

            raw_input = df_raw['inputValues'][i]
            for key in raw_input.keys():
                try:
                    total_dict[key].append(raw_input[key])
                except KeyError:
                    total_dict[key] = []
                    total_dict[key].append(raw_input[key])

            raw_output = df_raw['outputValues'][i]
            for key in raw_output.keys():
                if key not in self.factors:
                    continue
                try:
                    total_dict[key].append(self.keys[raw_output[key]])
                except KeyError:
                    total_dict[key] = []
                    total_dict[key].append(self.keys[raw_output[key]])

        # Aggregating across annotators
        if do_overlap:
            df_total = pd.DataFrame.from_dict(total_dict)
            a = df_total.groupby(['image_1'])
            total_dict = {}
            for name, group in a:
                for key in group.columns:
                    values = list(group[key])
                    new_values = aggregate(values)
                    try:
                        total_dict[key].append(new_values)
                    except KeyError:
                        total_dict[key] = []
                        total_dict[key].append(new_values)

        df_final = pd.DataFrame.from_dict(total_dict)
        if not keep_no_info:
            for factor in self.factors:
                df_final = df_final[df_final[factor] != self.keys['no_info']]
                df_final.reset_index(drop=True)

        return df_final
    # -----------------------------------------------------

    # -----------------------------------------------------
    def aggregate(self, df):
        assert isinstance(df, pd.DataFrame), "Aggregation requires parsed dataframe"

        # Aggregation function
        def calculate(inp):
            res = {}
            average_factors = []
            for factor in self.factors:
                factor_values = inp[factor]
                values = [elem for elem in factor_values if elem != -1]
                mean_factor_df = np.mean(values)
                average_factors.append(mean_factor_df)
                res[factor] = mean_factor_df
            res['score'] = np.mean(average_factors)
            return res

        # Bootstrapped estimation
        bootstrap_means = np.zeros(1000)
        for i in range(1000):
            bootstrap_sample = df.sample(n=len(df), replace=True)
            bootstrap_means[i] = calculate(bootstrap_sample)['score']
        confidence_interval = np.percentile(bootstrap_means, [0.5, 99.5])

        res = calculate(df)
        res['confidence_interval_99'] = confidence_interval

        return res
    # -----------------------------------------------------
