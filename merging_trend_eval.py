import argparse
from itertools import combinations
from typing import Optional, Dict

import torch
import wandb
from tqdm.auto import tqdm

from src.eval import eval_single_dataset
from src.task_vectors import (
    TaskVector,
    TaskVectorTopKZero,
    TaskVectorTopKInit,
    TaskVectorTopKKeep,
    TaskVectorMiddleKeep,
    TaskVectorRandomMask,
    TiesMerge,
)

zeroshot_acc = {
    "ViT-B-32": {
        "MNIST": 48.25,
        "RESISC45": 60.22,
        "DTD": 44.41,
        "GTSRB": 32.56,
        "SVHN": 31.61,
        "SUN397": 62.92,
        "EuroSAT": 45.15,
        "Cars": 59.64,
    },
    "ViT-B-16": {
        "MNIST": 51.80,
        "RESISC45": 66.35,
        "DTD": 44.68,
        "GTSRB": 43.34,
        "SVHN": 51.98,
        "SUN397": 65.22,
        "EuroSAT": 54.52,
        "Cars": 64.71,
    },
    "ViT-L-14": {
        "MNIST": 76.36,
        "RESISC45": 71.33,
        "DTD": 55.37,
        "GTSRB": 50.55,
        "SVHN": 58.45,
        "SUN397": 67.96,
        "EuroSAT": 61.63,
        "Cars": 77.94,
    },
}
finetuned_acc = {
    "ViT-B-32": {
        "MNIST": 99.69,
        "RESISC45": 96.11,
        "DTD": 79.41,
        "GTSRB": 98.73,
        "SVHN": 97.46,
        "SUN397": 74.98,
        "EuroSAT": 99.70,
        "Cars": 77.66,
    },
    "ViT-B-16": {
        "MNIST": 99.76,
        "RESISC45": 96.89,
        "DTD": 82.07,
        "GTSRB": 99.17,
        "SVHN": 97.86,
        "SUN397": 78.20,
        "EuroSAT": 99.70,
        "Cars": 86.79,
    },
    "ViT-L-14": {
        "MNIST": 99.69,
        "RESISC45": 97.37,
        "DTD": 84.15,
        "GTSRB": 99.24,
        "SVHN": 98.11,
        "SUN397": 81.96,
        "EuroSAT": 99.85,
        "Cars": 92.39,
    },
}


def main(args: argparse.Namespace):
    wandb.init(
        # Set the project where this run will be logged
        project="task-vector-addition",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{args.run_name}_alpha_{args.alpha}_beta_{args.beta}",
        # Track hyperparameters and run metadata
        config={
            "model": args.model,
            "alpha": args.alpha,
            "method": args.run_name,
            "beta": args.beta,
            "gamma": args.gamma,
            "eval_type": "partial" if args.eval_on_partial_datasets else "full",
        },
    )
    # build and load all the needed task vectors at once
    task_vectors_dict = None
    if args.run_name == "paper_implementation":
        task_vectors_dict = {
            dataset: TaskVector(
                args.pretrained_checkpoint, f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt"
            )
            for dataset in args.data_sets
        }

    elif args.run_name == "topk_zero":
        task_vectors_dict = {
            dataset: TaskVectorTopKZero(
                pretrained_checkpoint=args.pretrained_checkpoint,
                finetuned_checkpoint=f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt",
                top_k=args.beta,
            )
            for dataset in args.data_sets
        }
    elif args.run_name == "topk_init":
        task_vectors_dict = {
            dataset: TaskVectorTopKInit(
                pretrained_checkpoint=args.pretrained_checkpoint,
                finetuned_checkpoint=f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt",
                top_k=args.beta,
            )
            for dataset in args.data_sets
        }
    elif args.run_name == "topk_keep":
        task_vectors_dict = {
            dataset: TaskVectorTopKKeep(
                pretrained_checkpoint=args.pretrained_checkpoint,
                finetuned_checkpoint=f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt",
                top_k=args.beta,
            )
            for dataset in args.data_sets
        }
    elif args.run_name == "middle_keep":
        task_vectors_dict = {
            dataset: TaskVectorMiddleKeep(
                pretrained_checkpoint=args.pretrained_checkpoint,
                finetuned_checkpoint=f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt",
                top_k_keep=args.beta,
                top_k_remove=args.gamma,
            )
            for dataset in args.data_sets
        }
    elif args.run_name == "random":
        task_vectors_dict = {
            dataset: TaskVectorRandomMask(
                pretrained_checkpoint=args.pretrained_checkpoint,
                finetuned_checkpoint=f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt",
                keep=args.beta,
            )
            for dataset in args.data_sets
        }
    elif args.run_name == "ties":
        print("\U000026BD\U000026BD\U000026BD Initializing TIES merging.")
    else:
        raise ValueError("Unsupported method of task vectors.")

    base_depth = args.evaluation_depth if args.single_level_eval else 0
    for nb_datasets in tqdm(range(base_depth, args.evaluation_depth + 1)):
        global_average_acc = 0.0
        global_average_normalized_acc = 0.0
        global_imagenet_acc = 0.0
        nb_subset = 0
        for data_subsets in tqdm(combinations(args.data_sets, nb_datasets)):
            alpha = args.alpha
            average_acc = 0.0
            average_normalized_acc = 0.0

            if len(data_subsets) == 0:
                data_subsets = args.data_sets
                alpha = 0
            if args.run_name != "ties":
                assert task_vectors_dict is not None, "Was not able to populate the variable: task_vectors_dict"
                task_vectors = [task_vectors_dict[dataset] for dataset in data_subsets]
                task_vector_sum = sum(task_vectors)
                image_encoder = task_vector_sum.apply_to(args.pretrained_checkpoint, scaling_coef=alpha)
            else:
                ties_obj = TiesMerge(
                    pretrained_checkpoint=args.pretrained_checkpoint,
                    list_finetuned_checkpoints=[
                        f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt" for dataset in data_subsets
                    ],
                    top_k_keep=args.beta,
                )
                image_encoder = ties_obj.apply_to_pretrained(alpha=alpha)

            if not args.eval_on_imagenet_only:
                evaluation_dataset = data_subsets if args.eval_on_partial_datasets else args.data_sets
                for dataset in evaluation_dataset:
                    if len(evaluation_dataset) == 0:
                        try:
                            results = zeroshot_acc[args.model][dataset]
                        except KeyError:
                            results = eval_single_dataset(image_encoder, dataset, args)["top1"] * 100.0
                    else:
                        results = eval_single_dataset(image_encoder, dataset, args)["top1"] * 100.0

                    normalized_acc = (results / finetuned_acc[args.model][dataset]) * 100.0
                    wandb.log(
                        {
                            "current_task": dataset,
                            "individual_acc": results,
                            "individual_normalized_acc": normalized_acc,
                            "nb_task_vectors": nb_datasets,
                            "tasks": " ".join([str(t) for t in data_subsets]),
                        }
                    )
                    average_normalized_acc += normalized_acc
                    average_acc += results

                average_acc /= len(evaluation_dataset)
                average_normalized_acc /= len(evaluation_dataset)
                global_average_acc += average_acc
                global_average_normalized_acc += average_normalized_acc
                wandb.log(
                    {
                        "average_acc": average_acc,
                        "average_normalized_acc": average_normalized_acc,
                        "nb_task_vectors": nb_datasets,
                        "tasks": " ".join([str(t) for t in data_subsets]),
                    }
                )
                if args.eval_on_imagenet_also:
                    imagenet_results = eval_single_dataset(image_encoder, "ImageNet", args)["top1"] * 100.0
                    wandb.log(
                        {
                            "imagenet_acc": imagenet_results,
                            "nb_task_vectors": nb_datasets,
                            "tasks": " ".join([str(t) for t in data_subsets]),
                        }
                    )
                    global_imagenet_acc += imagenet_results
            else:
                # only evaluating on imagenet
                imagenet_results = eval_single_dataset(image_encoder, "ImageNet", args)["top1"] * 100.0
                wandb.log(
                    {
                        "imagenet_acc": imagenet_results,
                        "nb_task_vectors": nb_datasets,
                        "tasks": " ".join([str(t) for t in data_subsets]),
                    }
                )
                global_imagenet_acc += imagenet_results

            nb_subset += 1

        if not args.eval_on_imagenet_only:
            global_average_acc /= nb_subset
            global_average_normalized_acc /= nb_subset

            wandb.log(
                {
                    "global_average_acc": global_average_acc,
                    "global_average_normalized_acc": global_average_normalized_acc,
                    "nb_task_vectors": nb_datasets,
                }
            )
            if args.eval_on_imagenet_also:
                global_imagenet_acc /= nb_subset
                wandb.log(
                    {
                        "global_imagenet_acc": global_imagenet_acc,
                        "nb_task_vectors": nb_datasets,
                    }
                )
        else:
            global_imagenet_acc /= nb_subset
            wandb.log(
                {
                    "global_imagenet_acc": global_imagenet_acc,
                    "nb_task_vectors": nb_datasets,
                }
            )
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_location",
        help="Path to the root data.",
        default="~/data",
        type=str,
    )
    parser.add_argument(
        "--run_name",
        help="Optional name for the run.",
        type=str,
        default="paper_implementation",
        choices=["paper_implementation", "topk_zero", "topk_init", "topk_keep", "middle_keep", "random", "ties"],
    )
    parser.add_argument(
        "--checkpoint_path",
        help="Path to the directory that holds all the checkpoints.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--model",
        help="The type of model.",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14"],
        default="ViT-B-32",
        type=str,
    )
    parser.add_argument(
        "--data_sets",
        help="The name of the datasets used for evaluation",
        choices=["MNIST", "Cars", "RESISC45", "DTD", "GTSRB", "SVHN", "EuroSAT", "SUN397"],
        default="MNIST",
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--evaluation_depth",
        help="The depth refers to how many task vectors should be added for evaluation.",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--alpha",
        help="The value of alpha indicates the task vector multipliers.",
        default=0.4,
        type=float,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--beta",
        help="The removal value.",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--gamma",
        help="The removal value needed for the middle keep method.",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--single_level_eval",
        help="Indicates whether we want to evaluate on a single level basis or up to a level.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--eval_on_imagenet_only",
        help="Run all evaluations on imagenet data only.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--eval_on_imagenet_also",
        help="Run imagenet evaluation in addition to all other evaluations.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--eval_on_partial_datasets",
        help="If used, we evaluate only on the datasets relevant to the task vectors as opposed to all the datasets.",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.save = f"{args.checkpoint_path}/{args.model}"
    args.pretrained_checkpoint = f"{args.checkpoint_path}/{args.model}/zeroshot.pt"
    main(args=args)
