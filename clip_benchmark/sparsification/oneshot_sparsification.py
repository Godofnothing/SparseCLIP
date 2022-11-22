import os
import time
import math
import torch
import wandb
import numpy as np
import torch.nn.functional as F

from functools import partial
from contextlib import suppress
from typing import List, Optional
from collections import defaultdict
from torch.utils.data import Subset, DataLoader, TensorDataset

from sparseml.pytorch.optim import ScheduledModifierManager


__all__ = [
    "oneshot_sparsification"
]


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


# TODO add option of loading to drive?
@torch.no_grad()
def gather_inputs_and_outputs(model, loader, device, autocast=suppress):
    all_inputs = []
    all_outputs = []
    # collect outputs
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        with autocast():
            outputs = model(inputs)

        all_inputs.append(inputs.cpu())
        all_outputs.append(outputs.cpu())

    return (
        torch.cat(all_inputs, dim=0),
        torch.cat(all_outputs, dim=0)
    )



def oneshot_sparsification(
    # model
    model,
    # data
    dataset,
    # model props
    model_id: str,
    output_root: str,
    device: str,
    # few shot params
    fewshot_k: int,
    # sparseml params
    sparseml_recipe_path: str,
    # 
    create_grad_sampler: bool = False,
    create_calibration_loader: bool = True,
    # dataloader params
    num_grads: Optional[int] = None,
    grad_sampler_batch_size: Optional[int] = None,
    calibration_batch_size: Optional[int] = None,
    num_workers: Optional[int] = 1,
    amp: bool = True,
    loss: str = 'mse',
) -> None:
    output_dir = os.path.join(output_root, model_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # define loss function
    if loss == 'l2':
        loss_fn = F.mse_loss
    elif loss == 'l1':
        loss_fn = F.l1_loss
    elif loss == 'kl_div':
        loss_fn = partial(F.kl_div, log_target=True)
    else:
        raise ValueError('Unknown function.')

    # by default it is empty dict
    manager_kwargs = {}
    # create dedicated calibration loader
    if create_calibration_loader:
        assert calibration_batch_size is not None, \
            "One has to specify `calibration_batch_size` when requesting creation."
        # if fewshot make a subset of training data
        if fewshot_k > 0:
            fewshot_indices = get_fewshot_indices(dataset, fewshot_k)
            # make a subset of training dataset
            fewshot_dataset = Subset(dataset, fewshot_indices)
        else:
            fewshot_dataset = dataset

        # set autocast mode
        autocast = torch.cuda.amp.autocast if amp else suppress

        # create dataloaders
        dataloader = DataLoader(
            fewshot_dataset,
            batch_size=calibration_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        # collect dense model outputs
        model_inputs, model_outputs = gather_inputs_and_outputs(
            model,
            dataloader,
            device=device,
            autocast=autocast
        )

        calibration_dataset = TensorDataset(model_inputs, model_outputs)
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=calibration_batch_size or grad_sampler_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        # define calibration data_loader_builder
        def data_loader_builder(device=device, **kwargs):
            for input, target in calibration_loader:
                input, target = input.to(device), target.to(device)
                yield [input], {}, target

        manager_kwargs['calibration_sampler'] = {
            'data_loader_builder': data_loader_builder,
            'loss_fn': loss_fn,
        }

    if create_grad_sampler:
        # create dataloader 
        if num_grads is not None:
            feat_collection_indices = torch.randperm(len(dataset))[:num_grads * grad_sampler_batch_size]
            feat_collection_dataset = Subset(dataset, feat_collection_indices)
        else:
            feat_collection_dataset = dataset

        dataloader = DataLoader(
            feat_collection_dataset,
            batch_size=grad_sampler_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        # collect dense model outputs
        model_inputs, model_outputs = gather_inputs_and_outputs(
            model,
            dataloader,
            device=device,
            # grad sampler supports only fp32
            autocast=suppress
        )

        grad_sampler_dataset = TensorDataset(model_inputs, model_outputs)
        grad_sampler_loader = DataLoader(
            grad_sampler_dataset,
            batch_size=calibration_batch_size or grad_sampler_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        # define for WoofFisher/oViT pruner
        def data_loader_builder(device=device, **kwargs):
            while True:
                for input, target in grad_sampler_loader:
                    input, target = input.to(device), target.to(device)
                    yield [input], {}, target

        manager_kwargs['grad_sampler'] = {
            'data_loader_builder': data_loader_builder,
            'loss_fn': loss_fn,
        }

    # create manager
    manager = ScheduledModifierManager.from_yaml(sparseml_recipe_path)
    # apply recipe
    manager.apply(
        model,
        **manager_kwargs,
        finalize=True
    )
    # save resulting model
    torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint.pth'))
