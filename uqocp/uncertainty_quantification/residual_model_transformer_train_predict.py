# custom data parallel class
from torch.nn.parallel.scatter_gather import Scatter, _is_namedtuple, torch
from transformers.training_args import TrainingArguments
def custom_scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    batch_sizes = get_batch_idx(inputs["atoms_batch_idx"])
    num_batches_per_gpu = len(batch_sizes)/len(target_gpus)
    chunk_sizes = []
    current_chunk = 0
    current_num_batches = 0
    for batch_size in batch_sizes:
        if current_num_batches >= num_batches_per_gpu:
            chunk_sizes.append(current_chunk)
            current_chunk = 0
            current_num_batches = 0
        current_chunk += batch_size
        current_num_batches += 1
    chunk_sizes.append(current_chunk)
    if len(chunk_sizes) < len(target_gpus):
        target_gpus = target_gpus[:len(chunk_sizes)]

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res

import torch
torch.nn.parallel.scatter_gather.scatter = custom_scatter


from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DistilBertConfig, DistilBertModel, DistilBertForSequenceClassification, Trainer, TrainingArguments
import argparse
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch
import deepspeed
import inspect
from uuid import uuid4
import wandb
import time

import pprint
from transformers.modeling_outputs import SequenceClassifierOutput 
pp = pprint.PrettyPrinter(indent=4)


def get_parser():
    parser = argparse.ArgumentParser(description="traj transformer training script")

    # Add arguments
    # parser.add_argument(
    #     '--model',
    #     type=str,
    #     required=True,
    #     help='Model name'
    # )
    parser.add_argument(
        "--train_data",
        type=str,
        default="debug_data/id_adsorbml_results_w_pos_encodings.npz",
        help="npz path to training data",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="debug_data/ood_ads_adsorbml_results_w_pos_encodings.npz",
        help="npz path to validation data",
    )
    parser.add_argument(
        "--other_val_data",
        type=str,
        nargs="+",
        default=None,
        help="list of npz path(s) to additional validation data to use for logging (logs eval/other_loss)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="npz path to test data (only relevant for prediction)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        '--fp16',
        action="store_true",
        help="turn float precision 16 on, defaults to off",
        default=False,
    )
    parser.add_argument(
        '--no_fp16',
        dest="fp16",
        action="store_false",
        help="turn float precision 16 off, defaults to off",
        default=True,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="path to depespeed config file"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=3072,
        help="hidden dimension of transformer"
    )
    parser.add_argument(
        "--latent_limit",
        type=int,
        default=None,
        help="limit the length of the latent vector"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="default",
        help="path to save model to"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="path to load model from"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print configs",
        default=False,
    )
    parser.add_argument(
        "--task",
        type=str,
        help="'train' or 'predict'",
        default="train",
    )
    parser.add_argument(
        "--atoms_dim",
        type=str,
        help="what dimensional strategy to use for the atoms dimension: ['serial', 'batch']",
        default="batch",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="how often to save the model"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="how often to log on wandb"
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="which frames to sample for training/prediction"
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=49,
        help="dimension of latent vector"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=7,
        help="number of heads in transformer"
    )
    parser.add_argument(
        "--clip_labels",
        type=float,
        default=None,
        help="maximum error to train on, clip all error labels above this value"
    )

    return parser


def main(config):
    pp.pprint(config)

    train_data = None
    val_data = None
    test_data = None
    max_position_embeddings = 0

    # get data
    if config["atoms_dim"] == "serial":
        dataset_class = LabeledTrajDataset
        trainer_class = Trainer
        model_class = DistilBertForSequenceClassification
    elif config["atoms_dim"] == "batch":
        dataset_class = AtomsBatchDataset
        trainer_class = AtomsBatchTrainer
        model_class = AtomsBatchDistilBertForSequenceClassification
    else:
        raise ValueError(f"atoms_dim {config['atoms_dim']} not recognized")

    if config["train_data"] is not None:
        train_data = dataset_class(config["train_data"], fp16=config["fp16"], latent_limit=config["latent_limit"], frames=config["frames"], latent_dim=config["latent_dim"], clip_labels=config["clip_labels"])
        max_position_embeddings = max(train_data.max_latent, max_position_embeddings)
    if config["val_data"] is not None:
        val_data = dataset_class(config["val_data"], fp16=config["fp16"], latent_limit=config["latent_limit"], frames=config["frames"], latent_dim=config["latent_dim"], clip_labels=config["clip_labels"])
        max_position_embeddings = max(val_data.max_latent, max_position_embeddings)
    if config["test_data"] is not None:
        test_data = dataset_class(config["test_data"], fp16=config["fp16"], latent_limit=config["latent_limit"], frames=config["frames"], latent_dim=config["latent_dim"])
        max_position_embeddings = max(test_data.max_latent, max_position_embeddings)

    # init model
    model_config = DistilBertConfig(
        num_labels=1,
        max_position_embeddings=max_position_embeddings,
        dim=config["latent_dim"],
        n_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        torch_dtype="float16" if config["fp16"] else "float32",
        )
    if config["save"] is not None:
        save_path = config["save"]
        if "[unique]" in save_path:
            save_path = save_path.replace("[unique]", str(uuid4()))
            wandb.config.update({"save_path": save_path})
        model_config.save_pretrained(save_directory=os.path.join("./",save_path,"model_config"))

    if config["load"] is None:
        model = model_class(model_config)
    else:
        model = model_class.from_pretrained(config["load"])
        if config["task"] == train_data:
            model.train()

    if config["verbose"]:
        pp.pprint(model.config)

    # get training args
    training_args = TrainingArguments(
        os.path.join("./",save_path,"model_checkpoint"),
        evaluation_strategy="epoch",
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        push_to_hub=False,
        fp16=config["fp16"],
        num_train_epochs=config["epochs"],
        deepspeed=config["deepspeed"],
        local_rank=config["local_rank"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
    )
    if config["verbose"]:
        pp.pprint(training_args)

    if config["other_val_data"] is not None:
        name = config["val_data"].split("/")[-1].replace(".npz","")[:20]
        val_data = {
            name: val_data
        }
        for other_val_data_path in config["other_val_data"]:
            other_val_data = dataset_class(other_val_data_path, fp16=config["fp16"], latent_limit=config["latent_limit"], frames=config["frames"], latent_dim=config["latent_dim"], clip_labels=config["clip_labels"])
            name = other_val_data_path.split("/")[-1].replace(".npz","")[:20]
            val_data[name] = other_val_data

    # construct trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[EvalCallback()],
    )

    # run trainer
    if config["task"] == "train":
        trainer.train()
    elif config["task"] == "predict":
        output = trainer.predict(test_data)
        pp.pprint(output.metrics)
        np.savez_compressed(
            os.path.join("./",save_path,"transformer_results.npz"),
            err=output.label_ids,
            unc=output.predictions.flatten(),
        )
    else:
        raise ValueError(f"task {config['task']} not recognized")
    print('done')

        
class LabeledTrajDataset(Dataset):
    def __init__(
        self, 
        results_w_pos_encodings_file,
        fp16=False,
        latent_limit=None,
        frames=None,
        latent_dim=49,
        clip_labels=None
    ):
        self.latent_dim = latent_dim
        start_time = time.time()
        print(f"loading {results_w_pos_encodings_file}")
        with np.load(results_w_pos_encodings_file, allow_pickle=True) as data:
            self.latents = data["latents"]
            self.labels = data["labels"]
        if clip_labels is not None:
            self.labels = np.clip(self.labels, a_min=None, a_max=clip_labels)
        print(f"loaded {results_w_pos_encodings_file} ({time.time()-start_time} seconds))")

        if latent_limit is not None:
            for i in tqdm(range(len(self.latents)), "limiting latents"):
                if self.latents[i].shape[0] > latent_limit:
                    self.latents[i] = self.latents[i][-latent_limit:]

        self.max_latent = 0
        for latent in tqdm(self.latents, f"getting max latent"):
            if latent.shape[0] > self.max_latent:
                self.max_latent = latent.shape[0]
        self.max_latent = int(self.max_latent)
        print(f"max_latent: {self.max_latent}")

        self.float_str = "float16" if fp16 else "float32"
        # self.latents = [latent.astype(self.float_str) for latent in tqdm(self.latents, f"converting to {self.float_str}")]
        # self.labels = self.labels.astype(self.float_str)
        self.frames = np.array(frames) if frames is not None else None
        self.clip_labels = clip_labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx) -> tuple:
        item = {
            "inputs_embeds": self.latents[idx][:,2:],
            "labels": self.labels[idx],
            "attention_mask": np.ones(self.latents[idx].shape[0]),
        }
        if self.latents[idx].shape[0] < self.max_latent:
            item["inputs_embeds"] = np.concatenate([item["inputs_embeds"], np.zeros((self.max_latent-self.latents[idx].shape[0], self.latent_dim))])
            item["attention_mask"] = np.concatenate([item["attention_mask"], np.zeros(self.max_latent-self.latents[idx].shape[0])])

        for key, value in item.items():
            if hasattr(value, "astype"):
                item[key] = value.astype(self.float_str)
        return item


from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
class EvalCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for key, value in kwargs["metrics"].items():
            if "loss" in key:
                wandb.log({f"loss/{key}": value})
        return super().on_evaluate(args, state, control, **kwargs)



########## BATCH STUFF ##########

# atoms batch dataset class extension
class AtomsBatchDataset(LabeledTrajDataset):
    def __init__(
        self,
        results_w_pos_encodings_file,
        fp16=False,
        latent_limit=None,
        frames=None,
        latent_dim=49,
        clip_labels=None,
    ):
        super().__init__(
            results_w_pos_encodings_file,
            fp16,
            latent_limit,
            frames,
            latent_dim,
            clip_labels,
        )
        self.max_latent = int(np.max([np.max(latent[:,0]) for latent in tqdm(self.latents, "getting REAL max latents")]) + 1)
        print(f"REAL max_latent: {self.max_latent}")

    def __getitem__(self, idx) -> tuple:
        items = []
        inputs_embeds = self.latents[idx][:,2:]
        frame_ids = self.latents[idx][:,0]
        atom_ids = self.latents[idx][:,1]
        unique_atoms = np.unique(atom_ids)
        for atom_id in unique_atoms:
            new_item = {
                "inputs_embeds": inputs_embeds[atom_ids==atom_id],
                "labels": 0,
                "attention_mask": np.ones(inputs_embeds[atom_ids==atom_id].shape[0]),
                "atoms_batch_idx": idx,
            }
            if atom_id == 0:
                new_item["labels"] = self.labels[idx]

            input_length = new_item["inputs_embeds"].shape[0]
            if input_length < self.max_latent:
                new_item["inputs_embeds"] = np.concatenate([new_item["inputs_embeds"], np.zeros((self.max_latent-input_length, self.latent_dim))])
                new_item["attention_mask"] = np.concatenate([new_item["attention_mask"], np.zeros(self.max_latent-input_length)])

            if self.frames is not None:
                new_item["attention_mask"] = np.zeros_like(new_item["attention_mask"])
                new_item["attention_mask"][self.frames] = 1

            for key, value in new_item.items():
                if hasattr(value, "astype"):
                    new_item[key] = value.astype(self.float_str)
            items.append(new_item)
        return items
    
    def __getitems__(self, possibly_batched_index) -> list:
        list_of_lists = [self[idx] for idx in possibly_batched_index]
        data = [item for items in list_of_lists for item in items]
        return data

from transformers.trainer import (
    Dict,
    Union,
    Any,
    Tuple,
    Optional,
    List,
)

# atoms batch trainer class extension
class AtomsBatchTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        (loss, logits, labels) = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        if labels is not None:
            batch_idx = get_batch_idx(inputs["atoms_batch_idx"])
            labels = sum_over_batch_idx(batch_idx, labels)

            if labels.shape[0] != logits.shape[0]:
                raise ValueError(f"Custom scatter not working: labels.shape[0] != logits.shape[0]: {labels.shape[0]} != {logits.shape[0]}")

        return (loss, logits, labels)

from transformers.models.distilbert.modeling_distilbert import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, nn

# atoms batch distilbert model class extension
class AtomsBatchDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        atoms_batch_idx: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        batch_idx = get_batch_idx(atoms_batch_idx)
        logits = sum_over_batch_idx(batch_idx, logits)
        labels = sum_over_batch_idx(batch_idx, labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

# utility functions
def get_batch_idx(atoms_batch_idx):
    batch_idx = []
    count = 0
    for i, idx in enumerate(atoms_batch_idx):
        if idx != atoms_batch_idx[i-1] and i != 0:
            batch_idx.append(count)
            count = 0
        count += 1
    batch_idx.append(count)
    return batch_idx

def sum_over_batch_idx(batch_idx, input_tensor):
    return torch.stack([ten.sum(dim=0) for ten in torch.split(input_tensor, batch_idx)])

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)
