import os
import time
import math
from sklearn import datasets
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from contextlib import suppress
from typing import List, Optional
from collections import defaultdict
from torch.utils.data import Subset, DataLoader, TensorDataset

from sparseml.pytorch.optim import ScheduledModifierManager


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


'''
TODO add option of not using calibration dataset. 
TODO add OBS support.
'''


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
        # dataloader params
        calibration_batch_size: Optional[int] = None,
        num_workers: Optional[int] = 1,
        amp: bool = True,
        loss: str = 'mse',
) -> None:
    output_dir = os.path.join(output_root, model_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
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
        # TODO try different batch size?
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

    io_dataset = TensorDataset(model_inputs, model_outputs)
    io_loader = DataLoader(
        io_dataset,
        batch_size=calibration_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # define for OBC pruner
    def data_loader_builder(device=device, **kwargs):
        for input, target in io_loader:
            input, target = input.to(device), target.to(device)
            yield [input], {}, target

    # define loss function
    if loss == 'l2':
        loss_fn = F.mse_loss
    elif loss == 'l1':
        loss_fn = F.l1_loss
    elif loss == 'kl_div':
        loss_fn = partial(F.kl_div, log_target=True)
    else:
        raise ValueError('Unknown function.')

    manager_kwargs = dict(
        calibration_sampler={
            'data_loader_builder': data_loader_builder,
            'loss_fn': loss_fn,
        },
    )

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
