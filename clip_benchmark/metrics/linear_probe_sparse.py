import os
import time
import math
import torch
import wandb
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from typing import List
from contextlib import suppress
from collections import defaultdict
from timm.utils import AverageMeter
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

from sparseml.pytorch.optim import ScheduledModifierManager

from .linear_probe import Featurizer, FeatureDataset
from .zeroshot_classification import accuracy


def extract_features(
    model, 
    train_dataloader,
    val_dataloader, 
    feature_dir: str,
    device: str,
    sparsity: float,
    autocast=torch.cuda.amp.autocast,
):
    # create feature extractor
    featurizer = Featurizer(model).cuda()
    # now we have to cache the features
    devices = [x for x in range(torch.cuda.device_count())]
    featurizer = torch.nn.DataParallel(featurizer, device_ids=devices)

    for j, loader in enumerate([train_dataloader, val_dataloader]):
        save_str  = '_train' if j == 0 else '_val'
        # add sparsity suffix
        save_str += f'_sparsity={sparsity:.3f}'
        # skip in case features already exist
        if os.path.exists(os.path.join(feature_dir, f'features{save_str}.pt')):
            continue

        features = []
        targets = []
        num_batches_tracked = 0
        num_cached = 0
        with torch.no_grad():
            for images, target in tqdm(loader):
                images = images.to(device)

                with autocast():
                    feature = featurizer(images)
                
                features.append(feature.cpu())
                targets.append(target)

                num_batches_tracked += 1
                if (num_batches_tracked % 100) == 0:
                    features = torch.cat(features)
                    targets = torch.cat(targets)
                    
                    torch.save(features, os.path.join(feature_dir, f'features{save_str}_cache_{num_cached}.pt'))
                    torch.save(targets, os.path.join(feature_dir, f'targets{save_str}_cache_{num_cached}.pt'))
                    num_cached += 1
                    features = []
                    targets = []
        
        if len(features) > 0:
            features = torch.cat(features)
            targets = torch.cat(targets)
            torch.save(features, os.path.join(feature_dir, f'features{save_str}_cache_{num_cached}.pt'))
            torch.save(targets, os.path.join(feature_dir, f'targets{save_str}_cache_{num_cached}.pt'))
            num_cached += 1

        features = torch.load(os.path.join(feature_dir, f'features{save_str}_cache_0.pt'))
        targets = torch.load(os.path.join(feature_dir, f'targets{save_str}_cache_0.pt'))
        for k in range(1, num_cached):
            next_features = torch.load(os.path.join(feature_dir, f'features{save_str}_cache_{k}.pt'))
            next_targets = torch.load(os.path.join(feature_dir, f'targets{save_str}_cache_{k}.pt'))
            features = torch.cat((features, next_features))
            targets = torch.cat((targets, next_targets))

        for k in range(num_cached):
            os.remove(os.path.join(feature_dir, f'features{save_str}_cache_{k}.pt'))
            os.remove(os.path.join(feature_dir, f'targets{save_str}_cache_{k}.pt'))

        torch.save(features, os.path.join(feature_dir, f'features{save_str}.pt'))
        torch.save(targets, os.path.join(feature_dir, f'targets{save_str}.pt'))


def get_fewshot_indices(train_dataset, fewshot_k):
    # dict: class_id -> List[samples with class_id]
    class2ids = defaultdict(list)
    for i, class_id in enumerate(train_dataset.targets):
        class2ids[class_id].append(i)
    # take only k 
    for class_id in class2ids:
        class2ids[class_id] = \
            np.random.choice(class2ids[class_id], size=fewshot_k, replace=False)
    # gather all indices
    fewshot_indices = sum([ids.tolist() for _, ids in class2ids.items()], [])
    return fewshot_indices


def train_epoch(linear_probe, optimizer, criterion, loader, device, epoch: int, log_interval: int = 10, autocast=suppress) -> dict:
    linear_probe.train()
    # create meters
    loss_m = AverageMeter()
    lr_m   = AverageMeter()
    # start tick
    end = time.time()
    # train epoch
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        data_time = time.time() - end

        optimizer.zero_grad()
        with autocast():
            pred = linear_probe(inputs)
            loss = criterion(pred, targets)

        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()

        # update stats
        loss_m.update(loss.item(), len(inputs))
        lr_m.update(optimizer.param_groups[0]['lr'])

        if (i % log_interval) == 0:
            batches_per_epoch = len(loader)
            print(
                f"Train Epoch: {epoch} [{i:>4}/{batches_per_epoch:>4}]\t"
                f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t"
                f"LR {optimizer.param_groups[0]['lr']:.5f}"
            )

    return {'train/loss': loss_m.avg, 'lr': lr_m.avg}


@torch.no_grad()
def val_epoch(linear_probe, criterion, loader, device, epoch: int, autocast=suppress) -> dict:
    linear_probe.train()
    # create meters
    loss_m = AverageMeter()
    acc1_m = AverageMeter()
    acc5_m = AverageMeter()
    # train epoch
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        with autocast():
            pred = linear_probe(inputs)
            loss = criterion(pred, targets)

        # get accuracies
        acc1, acc5 = accuracy(pred.float(), targets.float(), topk=(1, 5))
        # update stats
        acc1_m.update(acc1, len(inputs))
        acc5_m.update(acc5, len(inputs))
        loss_m.update(loss.item(), len(inputs))

    print(
        f"Eval epoch: {epoch}\t"
        f"Acc1: {acc1:.3f}\tAcc5: {acc5:.3f}"
    )

    return {'val/loss': loss_m.avg, 'val/acc1': acc1_m.avg, 'val/acc5': acc5_m.avg}


def create_feat_dataloader(feature_dir: str, sparsity: float, batch_size: int, num_workers: int):
    # load saved features from drive
    train_features = torch.load(os.path.join(feature_dir, f'features_train_sparsity={sparsity:.3f}.pt'))
    train_targets  = torch.load(os.path.join(feature_dir, f'targets_train_sparsity={sparsity:.3f}.pt'))
    val_features   = torch.load(os.path.join(feature_dir, f'features_val_sparsity={sparsity:.3f}.pt'))
    val_targets    = torch.load(os.path.join(feature_dir, f'targets_val_sparsity={sparsity:.3f}.pt'))
    # create datasets
    train_feature_dset = FeatureDataset(train_features, train_targets)
    val_feature_dset   = FeatureDataset(val_features, val_targets)
    # create dataloaders
    train_feature_loader = DataLoader(
        train_feature_dset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
    )
    val_feature_loader = DataLoader(
        val_feature_dset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
    )

    return train_feature_loader, val_feature_loader


def create_lr_scheduler(optimizer, lr_schedule: str, epochs: int):
    if lr_schedule == 'linear':
        lr_schedule_fn = lambda epoch: 1 - epoch / epochs
    elif lr_schedule == 'cosine':
        lr_schedule_fn = lambda epoch: 0.5 * (1 + math.cos(math.pi * epoch / epochs))
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule_fn)
    return lr_scheduler


class CLIPClassifier(nn.Module):

    def __init__(self, image_encoder, linear_probe):
        super().__init__()
        self.image_encoder = image_encoder
        self.linear_probe  = linear_probe

    def forward(self, x: torch.Tensor):
        return self.linear_probe(self.visual(x))



def evaluate(
    # model
    model, 
    # data
    val_dataset,
    train_dataset, 
    collate_fn, 
    # few shot params
    fewshot_k: int,
    lr: float, 
    lr_schedule: str,
    # dataloader
    batch_size, 
    num_workers, 
    # training
    init_epochs: int,
    cycle_epochs: int, 
    # model props
    model_id, 
    feature_root, 
    device, 
    # sparseml params
    sparseml_recipe_path: str,
    sparsities: List[float],
    amp=True, 
    # one may want to finetune longer
    seed: int = 42,
    last_cycle_epochs: int = None,
    log_wandb: bool = False,
    log_interval: int = 20, 
) -> dict:
    # make dirs
    if not os.path.exists(feature_root):
        os.mkdir(feature_root)
    feature_dir = os.path.join(feature_root, model_id)
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
    
    # set autocast mode
    autocast = torch.cuda.amp.autocast if amp else suppress
    # by default last cycle uses the same number of epochs as the other
    last_cycle_epochs = last_cycle_epochs or cycle_epochs

    # if fewshot make a subset of training data
    if fewshot_k > 0:
        fewshot_indices = get_fewshot_indices(train_dataset, fewshot_k)
        # make a subset of training dataset
        train_dataset = Subset(train_dataset, fewshot_indices)

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # extract features and save to drive
    extract_features(
        model, 
        train_dataloader,
        val_dataloader,
        feature_dir,
        device=device,
        sparsity=0,
        autocast=autocast
    )
    # create feature dataloaders
    train_feature_loader, val_feature_loader = create_feat_dataloader(feature_dir, 0, batch_size, num_workers)
    # create linear probe
    train_features = train_feature_loader.dataset.features
    train_targets  = train_feature_loader.dataset.targets

    linear_probe = torch.nn.Linear(train_features[0].shape[0], train_targets.max().item() + 1).to(device)
    devices = [x for x in range(torch.cuda.device_count())]
    linear_probe = torch.nn.DataParallel(linear_probe, device_ids=devices)
    optimizer = torch.optim.AdamW(
        linear_probe.parameters(),
        lr=lr,
        weight_decay=0,
    )
    criterion = torch.nn.CrossEntropyLoss()
    # create scheduler
    lr_scheduler = create_lr_scheduler(optimizer, lr_schedule, init_epochs)

    last_epoch = 0
    # init training 
    for epoch in range(last_epoch, last_epoch + init_epochs):
        train_stats = train_epoch(
            linear_probe, 
            optimizer,
            criterion,
            loader=train_feature_loader, 
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            autocast=autocast
        )
        val_stats = val_epoch(
            linear_probe, 
            criterion,
            loader=val_feature_loader, 
            device=device,
            epoch=epoch,
            autocast=autocast
        )
        lr_scheduler.step()
        # log to wandb
        if log_wandb:
            wandb.log({**train_stats, **val_stats}, step=epoch)


    last_epoch = epoch + 1
    # pruning & retraining cycles
    for cycle_id, sparsity in enumerate(sparsities):
        # get number of epoch in current cycle
        cur_cycle_epochs = (cycle_epochs, last_cycle_epochs)[cycle_id == len(sparsities) - 1]
        # unfreeze model
        for param in model.visual.parameters():
            param.requires_grad = True
        # join linear probe and visual model
        classifier = CLIPClassifier(model.visual, linear_probe)
        # sparsify
        manager = ScheduledModifierManager.from_yaml(sparseml_recipe_path)
        # update manager
        manager.modifiers[0].init_sparsity  = sparsity
        manager.modifiers[0].final_sparsity = sparsity
        # apply one-shot pruning
        manager.apply(classifier)
        # freeze model again
        for param in model.visual.parameters():
            param.requires_grad = False
        # extract features and save to drive
        extract_features(
            model, 
            train_dataloader,
            val_dataloader,
            feature_dir,
            device=device,
            sparsity=sparsity,
            autocast=autocast
        )
        # load features from drive
        train_feature_loader, val_feature_loader = create_feat_dataloader(feature_dir, sparsity, batch_size, num_workers)
        # reset optimizer lr
        for param_groups in optimizer.param_groups:
            param_groups['lr'] = lr
        # create new scheduler
        lr_scheduler = create_lr_scheduler(optimizer, lr_schedule, cur_cycle_epochs)
        # train
        for epoch in range(last_epoch, last_epoch + cur_cycle_epochs):
            train_stats = train_epoch(
                linear_probe, 
                optimizer,
                criterion,
                loader=train_feature_loader, 
                device=device,
                epoch=epoch,
                log_interval=log_interval,
                autocast=autocast
            )
            val_stats = val_epoch(
                linear_probe, 
                criterion,
                loader=val_feature_loader, 
                device=device,
                epoch=epoch,
                autocast=autocast
            )
            lr_scheduler.step()
            if log_wandb:
                wandb.log({**train_stats, **val_stats}, step=epoch)

        last_epoch = epoch + 1

    return val_stats
