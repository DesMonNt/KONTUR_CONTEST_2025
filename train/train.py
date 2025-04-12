import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import log_softmax
from tqdm import tqdm
from .early_stopping import EarlyStopping


def get_warmup_scheduler(optimizer, warmup_steps=500):
    def lr_lambda(step):
        return min(1.0, step / warmup_steps)

    return LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0

    for features, targets, input_lengths, target_lengths in tqdm(loader, desc=f"[Epoch {epoch}] Train", leave=False):
        features, targets = features.to(device), targets.to(device)

        logits = model(features, input_lengths.to(device))
        log_probs = log_softmax(logits, dim=-1)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device, epoch):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for features, targets, input_lengths, target_lengths in tqdm(loader, desc=f"[Epoch {epoch}] Val", leave=False):
            features, targets = features.to(device), targets.to(device)

            logits = model(features, input_lengths.to(device))
            log_probs = log_softmax(logits, dim=-1)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            val_loss += loss.item()

    return val_loss / len(loader)


def train(model, train_loader, val_loader, num_epochs=100, lr=1e-3, warmup_steps=500, patience=10, device='cuda', save_path='models', log=False):
    os.makedirs(save_path, exist_ok=True)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_warmup_scheduler(optimizer, warmup_steps=warmup_steps)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    early_stopper = EarlyStopping(patience=patience, save_path=os.path.join(save_path, 'best_model.pt'))

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, epoch)
        early_stopper.step(val_loss, model)

        if log:
            print(f"[Epoch {epoch}] train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if early_stopper.early_stop:
            break

    return early_stopper.best_score
