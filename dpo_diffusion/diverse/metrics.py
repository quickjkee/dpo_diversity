import torch
import numpy as np

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score


def samples_metric(true, pred, n_boots=30):
    labels = set(true)
    class_weights = {}

    # Prepare weights for each class
    for label in labels:
        class_weights[label] = 0
        for el in true:
            if el == label:
                class_weights[label] += 1
        class_weights[label] = 1 / class_weights[label]
    sample_weights = [class_weights[item] for item in true]

    # Create dataset of indexes wrt weights
    idx = list(range(len(true)))
    dataset = TensorDataset(torch.tensor(idx))
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(true), replacement=True)

    # Bootstrapped accuracy
    accs = []
    for _ in range(n_boots):
        dl = DataLoader(dataset, sampler=sampler, batch_size=len(true))
        for item in dl:
            item = item[0]
            true_labels = [true[it.item()] for it in item]
            pred_labels = [pred[it.item()] for it in item]
            acc = accuracy_score(true_labels, pred_labels)
            accs.append(acc)

    return np.mean(accs), np.std(accs)