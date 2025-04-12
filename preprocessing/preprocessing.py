import os
import torch
import torchaudio
from tqdm import tqdm
from sklearn.model_selection import train_test_split

morse_dict = {
    'А': '.-', 'Б': '-...', 'В': '.--', 'Г': '--.', 'Д': '-..', 'Е': '.',
    'Ж': '...-', 'З': '--..', 'И': '..', 'Й': '.---', 'К': '-.-', 'Л': '.-..',
    'М': '--', 'Н': '-.', 'О': '---', 'П': '.--.', 'Р': '.-.', 'С': '...',
    'Т': '-', 'У': '..-', 'Ф': '..-.', 'Х': '....', 'Ц': '-.-.', 'Ч': '---.',
    'Ш': '----', 'Щ': '--.-', 'Ъ': '.--.-.', 'Ы': '-.--', 'Ь': '-..-',
    'Э': '..-..', 'Ю': '..--', 'Я': '.-.-',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    '#': '--.--', ' ': '/'
}
reverse_morse_dict = {v: k for k, v in morse_dict.items()}


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
        spec = preprocess_audio(os.path.join(audio_dir, fname), transform)
        label = encode_to_morse(labels_dict[fname])

        split_dir = val_dir if fname in val_files else train_dir
        torch.save((spec, label), os.path.join(split_dir, fname.replace('.opus', '.pt')))


def preprocess_audio(path, transform):
    waveform, sr = torchaudio.load(path)
    spec = transform(waveform).squeeze(0)

    return spec

def encode_to_morse(text):
    return ' '.join(morse_dict.get(ch.upper(), '?') for ch in text)

def decode_from_morse(morse_code):
    return ''.join(reverse_morse_dict.get(code, '') for code in morse_code.split())