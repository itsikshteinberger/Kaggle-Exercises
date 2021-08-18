import os
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch


class DigitSet(Dataset):
    def __init__(self, root_dir, csv_file, labels_file=None, transform=None):
        self.csv_file = pd.read_csv(os.path.join(root_dir, csv_file))
        self.transform = transform

        if labels_file:
            self.labels = pd.read_csv(os.path.join(root_dir, labels_file))["Label"]
        else:
            self.labels = self.csv_file["label"]
            self.csv_file = self.csv_file.iloc[:, 1:]

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, item):
        label = self.labels.iloc[item]
        image = self.csv_file.iloc[item, :]
        image = torch.tensor(image.values)
        image = torch.reshape(image, (1, 28, 28))
        return label, image
