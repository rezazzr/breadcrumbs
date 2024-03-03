from __future__ import annotations

import copy
from abc import ABC
from collections import OrderedDict
from typing import Optional, Dict, List

import torch

from src.utils import safe_load_state_dict


class TaskVectorABC(ABC):
    def __init__(
        self,
        pretrained_checkpoint=Optional[str],
        finetuned_checkpoint=Optional[str],
        vector=Optional[Dict[str, torch.Tensor]],
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = safe_load_state_dict(pretrained_checkpoint)
                finetuned_state_dict = safe_load_state_dict(finetuned_checkpoint)
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other: TaskVectorABC):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVectorABC(vector=new_vector)

    def __radd__(self, other: TaskVectorABC):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVectorABC(vector=new_vector)

    def apply_to(self, pretrained_checkpoint: str, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f"Warning: key {key} is present in the pretrained state dict but not in the task vector")
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


class TaskVector(TaskVectorABC):
    def __init__(
        self, pretrained_checkpoint=Optional[str], finetuned_checkpoint=Optional[str], vector=Dict[str, torch.Tensor]
    ):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)


class TaskVectorTopKZero(TaskVectorABC):
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, top_k: float = 0):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)
        self.top_k = top_k
        with torch.no_grad():
            for key, value in self.vector.items():
                self.vector[key] = self.mask(value)

    def mask(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 0:
            return tensor
        else:
            top_k_int = int(tensor.shape[-1] * self.top_k)
            _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
            mask = torch.ones(tensor.shape)
            mask.scatter_(len(tensor.shape) - 1, masked_indices, 0.0)

            return mask * tensor


class TaskVectorTopKInit(TaskVectorABC):
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, top_k: float = 0):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)
        self.top_k = top_k
        pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
        with torch.no_grad():
            for key, value in self.vector.items():
                self.vector[key] = self.mask_and_init(value, pretrained_state_dict[key])

    def mask_and_init(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
        if len(tensor_a.shape) == 0:
            return tensor_a
        else:
            top_k_int = int(tensor_a.shape[-1] * self.top_k)
            _, masked_indices = torch.topk(torch.abs(tensor_a), top_k_int)
            mask = torch.ones(tensor_a.shape)
            mask.scatter_(len(tensor_a.shape) - 1, masked_indices, 0.0)

            return mask * tensor_a + (~mask.bool()).int() * tensor_b


class TaskVectorTopKKeep(TaskVectorABC):
    def __init__(
        self,
        pretrained_checkpoint: Optional[str] = None,
        finetuned_checkpoint: Optional[str] = None,
        vector: Optional[Dict[str, torch.Tensor]] = None,
        top_k: float = 0,
    ):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)
        self.top_k = top_k
        with torch.no_grad():
            for key, value in self.vector.items():
                self.vector[key] = self.mask(value)

    def mask(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 0:
            return tensor
        else:
            top_k_int = int(tensor.shape[-1] * self.top_k)
            _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
            mask = torch.zeros(tensor.shape)
            mask.scatter_(len(tensor.shape) - 1, masked_indices, 1)

            return mask * tensor


class TaskVectorMiddleKeep(TaskVectorABC):
    def __init__(
        self,
        pretrained_checkpoint=None,
        finetuned_checkpoint=None,
        vector=None,
        top_k_keep: float = 0,
        top_k_remove: float = 0,
        remove_first: bool = True,
    ):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)
        self.top_k_keep = top_k_keep
        self.top_k_remove = top_k_remove
        with torch.no_grad():
            for key, value in self.vector.items():
                if remove_first:
                    self.vector[key] = self.mask_keep_top(self.mask_remove_top(value))
                else:
                    self.vector[key] = self.mask_remove_top(self.mask_keep_top(value))

    def mask_keep_top(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 0:
            return tensor
        else:
            top_k_int = int(tensor.shape[-1] * self.top_k_keep)
            _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
            mask = torch.zeros(tensor.shape)
            mask.scatter_(len(tensor.shape) - 1, masked_indices, 1)

            return mask * tensor

    def mask_remove_top(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 0:
            return tensor
        else:
            top_k_int = int(tensor.shape[-1] * self.top_k_remove)
            _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
            mask = torch.ones(tensor.shape)
            mask.scatter_(len(tensor.shape) - 1, masked_indices, 0.0)

            return mask * tensor


class TaskVectorRandomMask(TaskVectorABC):
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, keep: float = 0):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)
        self.keep = keep
        with torch.no_grad():
            for key, value in self.vector.items():
                self.vector[key] = self.mask_keep_random(value)

    def mask_keep_random(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 0:
            return tensor
        else:
            mask: torch.Tensor = torch.rand(tensor.shape) < self.keep
            return mask * tensor


class TiesMerge:
    def __init__(self, pretrained_checkpoint: str, list_finetuned_checkpoints: List[str], top_k_keep: float = 0.2):
        self.top_k_keep = top_k_keep
        self.path_pretrained_checkpoint = pretrained_checkpoint
        # load the models
        assert pretrained_checkpoint is not None and len(list_finetuned_checkpoints) > 0
        with torch.no_grad():
            print("--> Loading checkpoints.")
            self.pretrained_state_dict = safe_load_state_dict(pretrained_checkpoint)
            list_finetuned_state_dict = [
                safe_load_state_dict(finetuned_checkpoint) for finetuned_checkpoint in list_finetuned_checkpoints
            ]

            print("--> Flattening out the checkpoints.")
            stacked_flat_finetuned = torch.vstack(
                [
                    TiesMerge.state_dict_to_vector(finetuned_state_dict)
                    for finetuned_state_dict in list_finetuned_state_dict
                ]
            )
            self.flat_pretrained = TiesMerge.state_dict_to_vector(self.pretrained_state_dict)

            # create task vector i.e. (Finetuned - Pretrained)
            flat_task_vectors = stacked_flat_finetuned - self.flat_pretrained

            # check if the vectorized state dicts can be converted back to the original state dicts
            # convert back the flat task vectors to state dict and see if the original and converted ones are equal
            assert TiesMerge.check_state_dicts_equal(
                TiesMerge.vector_to_state_dict(self.flat_pretrained, self.pretrained_state_dict),
                self.pretrained_state_dict,
            )
            assert all(
                [
                    TiesMerge.check_state_dicts_equal(
                        state_dict1=TiesMerge.vector_to_state_dict(flat_finetuned, self.pretrained_state_dict),
                        state_dict2=fine_tuned_state_dict,
                    )
                    for fine_tuned_state_dict, flat_finetuned in zip(list_finetuned_state_dict, stacked_flat_finetuned)
                ]
            )
            print("--> Initiating TIES Merger.")
            self.flat_merged_task_vectors = TiesMerge.apply_ties_on_task_vectors(
                flat_task_vectors=flat_task_vectors, top_k_keep=self.top_k_keep
            )

    def apply_to_pretrained(self, alpha: float = 1) -> torch.nn.Module:
        flat_integration: torch.Tensor = self.flat_pretrained + alpha * self.flat_merged_task_vectors
        merged_state_dict = TiesMerge.vector_to_state_dict(
            vector=flat_integration, state_dict=self.pretrained_state_dict
        )
        pretrained_model: torch.nn.Module = torch.load(self.path_pretrained_checkpoint)
        pretrained_model.load_state_dict(merged_state_dict, strict=False)
        return pretrained_model

    @staticmethod
    def state_dict_to_vector(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        shared_state_dict = copy.deepcopy(state_dict)
        for key, value in shared_state_dict.items():
            if value.dtype in [torch.int64, torch.uint8]:
                del shared_state_dict[key]

        sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
        )

    @staticmethod
    def vector_to_state_dict(vector: torch.Tensor, state_dict: Dict[str, torch.Tensor]):
        # create a reference dict to define the order of the vector
        reference_dict = copy.deepcopy(state_dict)
        ignored_items = {}
        for key, value in reference_dict.items():
            if value.dtype in [torch.int64, torch.uint8]:
                ignored_items[key] = value
                del reference_dict[key]

        sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

        # create a shared state dict using the reference dict
        torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())
        # add back the ignored items
        for key in ignored_items:
            sorted_reference_dict[key] = ignored_items[key]

        return sorted_reference_dict

    @staticmethod
    def check_state_dicts_equal(state_dict1: Dict[str, torch.Tensor], state_dict2: Dict[str, torch.Tensor]) -> bool:
        if set(state_dict1.keys()) != set(state_dict2.keys()):
            return False

        for key in state_dict1.keys():
            if not torch.equal(state_dict1[key], state_dict2[key]):
                return False

        return True

    @staticmethod
    def keep_top_k_for_every_row(tensor: torch.Tensor, top_k_keep: float = 0.2) -> torch.Tensor:
        # the `tensor` must have multiple rows. The algorithm will mask out the small values in each row and keep
        # the top k values.
        original_shape = tensor.shape
        if tensor.dim() == 1:
            # this turns the `tensor` to a single row matrix, so we can operate on it as usual.
            tensor = tensor.unsqueeze(0)

        rows, columns = tensor.shape
        top_k_keep = columns - int(columns * top_k_keep)  # Keep top k elements instead of bottom k elements

        # Find the k-th smallest element by magnitude for each row
        kth_values, _ = tensor.abs().kthvalue(top_k_keep, dim=1, keepdim=True)
        # Create a mask tensor with True for the top k elements in each row
        mask: torch.Tensor = tensor.abs() >= kth_values
        final_mask = mask.squeeze() if original_shape == tensor.squeeze().shape else mask

        return tensor * final_mask

    @staticmethod
    def resolve_sign(tensor: torch.Tensor) -> torch.Tensor:
        signs_to_multiply = torch.sign(tensor.sum(dim=0))
        # resolve zero signs
        majority_sign = torch.sign(signs_to_multiply.sum())
        signs_to_multiply[signs_to_multiply == 0] = majority_sign

        return signs_to_multiply

    @staticmethod
    def disjoint_mean_merge(tensor: torch.Tensor, signs_to_multiply: torch.Tensor) -> torch.Tensor:
        # If sign is provided then we select the corresponding entries and aggregate.
        if signs_to_multiply is not None:
            rows_to_keep = torch.where(signs_to_multiply.unsqueeze(0) > 0, tensor > 0, tensor < 0)
            selected_entries = tensor * rows_to_keep
        # Else we select all non-zero entries and aggregate.
        else:
            rows_to_keep = tensor != 0
            selected_entries = tensor * rows_to_keep

        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggregates = torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
        return disjoint_aggregates

    @staticmethod
    def apply_ties_on_task_vectors(flat_task_vectors: torch.Tensor, top_k_keep: float):
        copy_flat_task_vectors = flat_task_vectors.clone()
        # mask out the bottom values i.e. keep top k values
        print(f"[TIES STEP 1]: Masking the lower values, keeping only {top_k_keep * 100}%.")
        masked_flat_task_vectors = TiesMerge.keep_top_k_for_every_row(tensor=flat_task_vectors, top_k_keep=top_k_keep)

        print("[TIES STEP 2]: Resolving the signs.")
        final_signs = TiesMerge.resolve_sign(tensor=masked_flat_task_vectors)
        assert final_signs is not None

        print("[TIES STEP 3]: Disjoint mean merge.")
        flat_merged_task_vectors = TiesMerge.disjoint_mean_merge(
            tensor=masked_flat_task_vectors, signs_to_multiply=final_signs
        )
        return flat_merged_task_vectors
