import os
import torch
import torchaudio
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def preprocess_dataset(audio_dir, save_dir, labels_dict, transform, val_split=0.15, seed=42):
    os.makedirs(save_dir, exist_ok=True)

    train_dir = os.path.join(save_dir, 'train')
    val_dir = os.path.join(save_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    files = [f for f in os.listdir(audio_dir) if f.endswith('.opus')]
    files = [f for f in files if int(f.replace('.opus', '')) <= 30000]

    train_files, val_files = train_test_split(files, test_size=val_split, random_state=seed)
    val_files = set(val_files)

    for fname in tqdm(files, desc=f"Preprocessing {audio_dir}"):
        mel = preprocess_audio(os.path.join(audio_dir, fname), transform)
        label = labels_dict[fname]

        split_dir = val_dir if fname in val_files else train_dir
        torch.save((mel, label), os.path.join(split_dir, fname.replace('.opus', '.pt')))


def preprocess_audio(path, transform):
    waveform, sr = torchaudio.load(path)
    mel = transform(waveform).squeeze(0)

    return mel