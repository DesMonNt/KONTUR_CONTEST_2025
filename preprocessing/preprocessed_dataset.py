from torch.utils.data import Dataset
import torch
import os

class PreprocessedDataset(Dataset):
    def __init__(self, dataset_path, char2idx, transform=None):
        self.paths = sorted([
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path) if f.endswith('.pt')
        ])
        self.char2idx = char2idx
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        spec, label = torch.load(self.paths[idx])

        if self.transform:
            spec = self.transform(spec)

        input_length = spec.shape[-1]
        label = torch.tensor([self.char2idx[c] for c in label], dtype=torch.long)
        target_length = len(label)

        return spec, label, input_length, target_length
