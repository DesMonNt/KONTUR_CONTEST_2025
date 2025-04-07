import torch

def normalize_nonzero(spec):
    energy = spec.mean(dim=1).squeeze(0)
    mask = energy > -80

    if mask.sum() == 0:
        return spec

    active = spec[:, :, mask]

    mean = active.mean()
    std = active.std()

    return (spec - mean) / (std + 1e-3)

def build_vocab(labels):
    unique_chars = sorted(set(''.join(labels)))
    char2idx = {c: i + 1 for i, c in enumerate(unique_chars)}
    idx2char = {i: c for c, i in char2idx.items()}

    return char2idx, idx2char

def greedy_decode(log_probs, input_lengths, idx2char):
    decoded_batch = []
    max_probs = torch.argmax(log_probs, dim=-1)
    max_probs = max_probs.transpose(0, 1)

    for seq, length in zip(max_probs, input_lengths):
        seq = seq[:length]
        decoded = []
        prev = -1

        for idx in seq:
            idx = idx.item()

            if idx != prev and idx != 0:
                decoded.append(idx2char.get(idx, '?'))

            prev = idx

        decoded_batch.append("".join(decoded))

    return decoded_batch