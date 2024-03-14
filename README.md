# Breadcrumbs

## Overview

Welcome to the Breadcrumbs repository, featuring the `TaskVectorMiddleKeep` method (this is the Breadcrumbs method in the code, specifically `src/task_vectors.py` file).

## Usage Example

To demonstrate the usage of the `TaskVectorMiddleKeep` method, consider the following example:

### Checkpoints
Checkpoints for CLIP ViT-B/32, ViT-B/16 and ViT-L/14 are available on the link below, including fine-tuned checkpoints on eight downstream tasks: Stanford Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397 and SVHN.

[Download here](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw?usp=share_link)

```python
import torch
import argparse
from task_vectors import TaskVectorMiddleKeep
from eval import eval_single_dataset

# Config
datasets = ['MNIST', 'RESISC45']
model = 'ViT-L-14'

# Create argparse.Namespace for configuration
args = argparse.Namespace(
    data_location='/path/to/data',
    model=model,
    save=f'checkpoints/{model}',
    pretrained_checkpoint=f'checkpoints/{model}/zeroshot.pt',
    evaluation_depth=1,
    alpha=0.4,
    batch_size=1024,
    beta=0.2,
    gamma=0.1,
    single_level_eval=False,
    eval_on_imagenet_only=False,
    eval_on_imagenet_also=False,
    eval_on_partial_datasets=False,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

# Create the task vectors using TaskVectorMiddleKeep
task_vectors = {
    dataset: TaskVectorMiddleKeep(
        pretrained_checkpoint=args.pretrained_checkpoint,
        finetuned_checkpoint=f'checkpoints/{model}/{dataset}/finetuned.pt',
        top_k_keep=args.beta,
        top_k_remove=args.gamma,
    )
    for dataset in datasets
}

# Sum the task vectors
task_vector_sum = sum(task_vectors.values())

# Apply the resulting task vector
image_encoder = task_vector_sum.apply_to(args.pretrained_checkpoint, scaling_coef=args.alpha)

# Evaluate on specified datasets
for dataset in datasets:
    eval_single_dataset(image_encoder, dataset, args)
```

Make sure to adjust the configuration parameters as needed for your specific use case. This example demonstrates the creation of `TaskVectorMiddleKeep` instances for each dataset, combining them, applying the resulting task vector to the pretrained model, and evaluating the model on the specified datasets.

Feel free to explore and experiment with different configurations to observe the impact of the `TaskVectorMiddleKeep` method in other settings.