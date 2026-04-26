# from flautim.pytorch.Dataset import Dataset
from torch.utils.data import Dataset
import torch
import copy

class FLLMDataset(Dataset):

    def __init__(self, input_ids, attention_mask, labels, **kwargs):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def train(self) -> Dataset:

        return copy.deepcopy(self)

    def validation(self) -> Dataset:
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
                "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
                "labels": torch.tensor(self.labels[idx], dtype=torch.float)
                }