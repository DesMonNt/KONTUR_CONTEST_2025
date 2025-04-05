import os
import torch
import torch.nn as nn
from early_stopping import EarlyStopping
from torch.optim import Adam
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    for features, targets, input_lengths, target_lengths in tqdm(loader, desc=f"[Epoch {epoch}] Train", leave=False):
        features, targets = features.to(device), targets.to(device)

        logits = model(features)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device, epoch):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for features, targets, input_lengths, target_lengths in tqdm(loader, desc=f"[Epoch {epoch}] Val", leave=False):
            features, targets = features.to(device), targets.to(device)

            logits = model(features)
            log_probs = nn.functional.log_softmax(logits, dim=-1)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            val_loss += loss.item()

    return val_loss / len(loader)


def train(model, train_loader, val_loader, num_epochs=100, lr=1e-3, device='cuda', save_path='models'):
    os.makedirs(save_path, exist_ok=True)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    early_stopper = EarlyStopping(patience=10, save_path=os.path.join(save_path, 'best_model.pt'))

    for epoch in range(1, num_epochs + 1):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, epoch)

        early_stopper.step(val_loss, model)

        if early_stopper.early_stop:
            break