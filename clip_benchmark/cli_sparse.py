"""Console script for clip_benchmark."""

import sys
import json
import torch
import wandb
import argparse
import open_clip

from datasets.builder import build_dataset, get_dataset_collate_fn
from clip_benchmark.sparsification import sparse_linear_probing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cifar10", help="Dataset to use for the benchmark")
    parser.add_argument('--split', type=str, default="test", help="Dataset split to use")
    parser.add_argument('--model', type=str, default="ViT-B-32-quickgelu", help="Model architecture to use from OpenCLIP")
    parser.add_argument('--pretrained', type=str, default="laion400m_e32", help="Model checkpoint name to use from OpenCLIP")
    parser.add_argument('--task', type=str, default="zeroshot_classification", choices=["linear_probe"])
    parser.add_argument('--amp', default=True, action="store_true", help="whether to use mixed precision")
    parser.add_argument('--num_workers', default=4, type=int)
    # Few shot params
    parser.add_argument('--fewshot_k', default=-1, type=int, help="for linear probe, how many shots. -1 = whole dataset.")
    parser.add_argument('--fewshot_lr', default=0.1, type=float, help="for linear probe, what is the learning rate.")
    parser.add_argument('--lr_schedule', default='cosine', type=str, choices=['linear', 'cosine'], help="Learning rate schedule.")
    parser.add_argument('--init_epochs', default=10, type=int, help="number of epochs for initial linear probing.")
    parser.add_argument('--cycle_epochs', default=5, type=int, help="number of epochs for pruning cycle with linear probing.")
    parser.add_argument('--last_cycle_epochs', default=None, type=int, help="number of epochs for last pruning cycle with linear probing.")
    # Loading params
    parser.add_argument('--seed', default=0, type=int, help="random seed.")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset_root', default="root", type=str, help="dataset root folder where the datasets are downloaded.")
    parser.add_argument('--feature_root', default="features", type=str, help="feature root folder where the features are stored.")
    parser.add_argument('--annotation_file', default="", type=str, help="text annotation file for retrieval datasets. Only needed  for when `--task` is `zeroshot_retrieval`.")
    parser.add_argument('--output', default="result.json", type=str, help="output file where to dump the metrics")
    # SparseML parameters
    parser.add_argument('--sparseml_recipe_path', type=str, required=True, help="Path to SparseML recipe used as a template")
    parser.add_argument('--sparsities', type=float, nargs="+", help="List of intermediate sparsities")
    parser.add_argument('--create_calbiration_dataset', action='store_true', help="Whether to create additional dataset.")
    parser.add_argument('--calbiration_k', default=-1, type=int, help="number of shots for calibration dataset.")
    # Logging parameters
    parser.add_argument('--verbose', default=False, action="store_true", help="verbose mode")
    parser.add_argument('--log_wandb', default=False, action="store_true", help="Log to W&B")
    parser.add_argument('--log_interval', default=20, type=int, help="Logging interval")
    args = parser.parse_args()
    run(args)
    
def run(args):
    """Console script for clip_benchmark."""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # set seed.
    torch.manual_seed(args.seed)
    # init W&B logger
    if args.log_wandb:
        wandb.init(config=args)

    model, _, transform = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(args.device)
    val_dataset = build_dataset(
        dataset_name=args.dataset, 
        root=args.dataset_root, 
        transform=transform, 
        split=args.split, 
        annotation_file=args.annotation_file,
        download=True,
    )
    collate_fn = get_dataset_collate_fn(args.dataset)
    if args.verbose:
        print(f"Dataset size: {len(val_dataset)}")
        print(f"Dataset split: {args.split}")
        print(f"Dataset classes: {val_dataset.classes}")
        print(f"Dataset number of classes: {len(val_dataset.classes)}")

    if args.task == "linear_probe":
        # we also need the train split for linear probing.
        train_dataset = build_dataset(
            dataset_name=args.dataset, 
            root=args.dataset_root, 
            transform=transform, 
            split='train', 
            annotation_file=args.annotation_file,
            download=True,
        )

        metrics = sparse_linear_probing.evaluate(
            model,
            val_dataset=val_dataset,
            train_dataset=train_dataset,
            # SparseML params
            sparseml_recipe_path=args.sparseml_recipe_path,
            sparsities=args.sparsities,
            # 
            collate_fn=collate_fn,
            fewshot_k=args.fewshot_k,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.fewshot_lr,
            lr_schedule=args.lr_schedule,
            init_epochs=args.init_epochs,
            cycle_epochs=args.cycle_epochs,
            last_cycle_epochs=args.last_cycle_epochs,
            model_id=(args.model + '-' + args.pretrained + '-' + args.dataset).replace('/', '_'),
            seed=args.seed,
            feature_root=args.feature_root,
            device=args.device, 
            amp=args.amp,
            log_interval=args.log_interval,
            log_wandb=args.log_wandb
        )
    else:
        raise ValueError("Unsupported task: {}. task should `zeroshot_classification` or `zeroshot_retrieval`".format(args.task))

    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": args.task,
        "metrics": metrics
    }

    with open(args.output, "w") as f:
        json.dump(dump, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
