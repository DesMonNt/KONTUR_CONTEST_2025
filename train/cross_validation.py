from sklearn.model_selection import KFold
import torch
import copy

def cross_val_scores(model_instance, dataset, train_fn, collate_fn, num_folds=5, device='cuda', **train_kwargs):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    losses = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        train_subset = torch.utils.data.Subset(dataset, train_ids)
        val_subset = torch.utils.data.Subset(dataset, val_ids)

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=256, shuffle=True, collate_fn=collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=256, shuffle=False, collate_fn=collate_fn
        )

        model = copy.deepcopy(model_instance)

        val_loss = train_fn(model, train_loader, val_loader, device=device, **train_kwargs)
        losses.append(val_loss)

    return losses
