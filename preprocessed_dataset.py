from torch.utils.data import Dataset
import torch
import os

class PreprocessedDataset(Dataset):
    def __init__(self, dataset_path, char2idx):
        self.paths = sorted([
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path) if f.endswith('.pt')
        ])
        self.char2idx = char2idx

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mel, label = torch.load(self.paths[idx])
        input_length = mel.shape[-1]
        label_tensor = torch.tensor([self.char2idx[c] for c in label], dtype=torch.long)
        target_length = len(label_tensor)

        return mel, label_tensor, input_length, target_length
