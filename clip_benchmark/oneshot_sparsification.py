"""Console script for clip_benchmark."""

import sys
import torch
import argparse
import open_clip
import numpy as np

from datasets.builder import build_dataset
from sparsification import oneshot_sparsification
from utils.model_specific import fix_attention_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cifar10", 
                        help="Dataset to use for the benchmark")
    parser.add_argument('--dataset_root', default="root", type=str, 
                        help="dataset root folder where the datasets are downloaded.")
    parser.add_argument('--model', type=str, default="ViT-B-32-quickgelu", 
                        help="Model architecture to use from OpenCLIP")
    parser.add_argument('--pretrained', type=str, default="laion400m_e32", 
                        help="Model checkpoint name to use from OpenCLIP")
    parser.add_argument('--amp', default=False, action="store_true", 
                        help="Whether to use mixed precision")
    parser.add_argument('--num_workers', default=4, type=int)
    # Few shot params
    parser.add_argument('--fewshot_k', default=-1, type=int, 
                        help="How many shots are taken for calibration. -1 = whole dataset.")
    # Loading params
    parser.add_argument('--create_grad_sampler', action='store_true', 
                        help='Whether to create GradSampler loader.')
    parser.add_argument('--create_calibration_loader', action='store_true', 
                        help='Whether to create calibration loader (for OBC/FastOBC).')
    parser.add_argument('--calibration_batch_size', default=64, type=int)
    parser.add_argument('--grad_sampler_batch_size', default=64, type=int)
    # Sparsification parameters
    parser.add_argument('--sparseml_recipe_path', type=str, required=True, 
                        help="Path to SparseML recipe")

    parser.add_argument('--loss', default='l2', choices=['l1', 'l2', 'kl_div'], type=str, 
                        help='The loss used to measure discrepancy between dense and sparse features.')
    # Misc params
    parser.add_argument('--fix_attention_layer', action='store_true', 
                        help='Set True to make attention layer SparseML friendly.')
    parser.add_argument('--seed', default=0, type=int, 
                        help="Random seed.")
    parser.add_argument('--output_root', default="oneshot_outputs", type=str, 
                        help="Outputs where pruned models are stored.")

    args = parser.parse_args()
    run(args)
    
def run(args):
    """Console script for clip_benchmark."""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # set seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # create dataset
    model, _, transform = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    # we need only visual part
    model = model.visual.to(args.device)
    # fix model
    if args.fix_attention_layer:
        model = fix_attention_layer(model)

    train_dataset = build_dataset(
        dataset_name=args.dataset, 
        root=args.dataset_root, 
        transform=transform, 
        split='train', 
        annotation_file=None,
        download=False,
    )

    oneshot_sparsification.oneshot_sparsification(
        # we need only vision backbone
        model,
        dataset=train_dataset,
        model_id=(args.model + '-' + args.pretrained + '-' + args.dataset).replace('/', '_'),
        # SparseML params
        sparseml_recipe_path=args.sparseml_recipe_path,
        # Fewshot params
        fewshot_k=args.fewshot_k,
        # Dataloader params
        create_calibration_loader=args.create_calibration_loader,
        create_grad_sampler=args.create_grad_sampler,
        calibration_batch_size=args.calibration_batch_size,
        grad_sampler_batch_size=args.grad_sampler_batch_size,
        # Misc params
        num_workers=args.num_workers,
        output_root=args.output_root,
        device=args.device, 
        amp=args.amp,
        loss=args.loss
    )

    print('Sparsification completed!')



if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
