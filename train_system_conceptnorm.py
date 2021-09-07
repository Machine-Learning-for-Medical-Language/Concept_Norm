# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on clinical NLP tasks"""

import dataclasses
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from os.path import basename, dirname
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import torch
from filelock import FileLock
from torch import nn
from torch.utils.data.dataset import Dataset
from transformers import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, AutoConfig, AutoModelForSequenceClassification, \
    AutoTokenizer, DataCollatorWithPadding, EvalPrediction, HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers.data.metrics import acc_and_f1
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from transformers.training_args import EvaluationStrategy, IntervalStrategy

from cnlp_data_Bert_conceptnorm import ClinicalNlpDataset, DataTrainingArguments
from cnlp_processors import cnlp_compute_metrics, cnlp_output_modes, cnlp_processors, tagging
from CnlpBert_conceptnorm import CnlpBertForConceptNorm

logger = logging.getLogger(__name__)

InputDataClass = NewType("InputDataClass", Any)


def default_data_collator(
        features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "st_labels" in first and first["st_labels"] is not None:
        st_labels = first["st_labels"].item() if isinstance(
            first["st_labels"], torch.Tensor) else first["st_labels"]
        dtype = torch.long if isinstance(st_labels, int) else torch.float
        batch["st_labels"] = torch.tensor([f["st_labels"] for f in features],
                                          dtype=dtype)
    if "concept_labels" in first and first["concept_labels"] is not None:
        concept_labels = first["concept_labels"].item() if isinstance(
            first["concept_labels"], torch.Tensor) else first["concept_labels"]
        dtype = torch.long if isinstance(concept_labels, int) else torch.float
        batch["concept_labels"] = torch.tensor(
            [f["concept_labels"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("concept_labels",
                     "st_labels") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


@dataclass
class CnlpTrainingArguments(TrainingArguments):
    """
    Additional arguments specific to this class
    """
    # evals_per_epoch: Optional[int] = field(
    #     default=-1,
    #     metadata={
    #         "help":
    #         "Number of times to evaluate and possibly save model per training epoch (allows for a lazy kind of early stopping)"
    #     })

    # train_batch_size: int = field(
    #     default=32,
    #     metadata={
    #         "help":
    #         "The maximum total input sequence length after tokenization. Sequences longer "
    #         "than this will be truncated, sequences shorter will be padded."
    #     },
    # )

    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The learning rate of the optimizer"},
    )

    num_train_epochs: int = field(
        default=5,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    label_names: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "A space-separated list of labels"})

    eval_accumulation_steps: Optional[int] = field(
        default=32,
        metadata={
            "help":
            "Number of predictions steps to accumulate before moving the tensors to the CPU."
        },
    )

    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={
            "help":
            "Whether or not to load the best model found during training at the end of training."
        },
    )

    concept_embeddings_pre: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to load the pre-trained concept embeddings"
        },
    )

    # metric_for_best_model: Optional[str] = field(
    #     default="acc",
    #     metadata={
    #         "help": "The metric to use to compare two different models."
    #     })


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from s3"
        })
    layer: Optional[int] = field(
        default=-1,
        metadata={"help": "Which layer's CLS ('<s>') token to use"})
    token: bool = field(
        default=False,
        metadata={
            "help":
            "Classify over an actual token rather than the [CLS] ('<s>') token -- requires that the tokens to be classified are surrounded by <e>/</e> tokens"
        })
    freeze: bool = field(
        default=False,
        metadata={
            "help":
            "Freeze the encoder layers and only train the layer between the encoder and classification architecture. Probably works best with --token flag since [CLS] may not be well-trained for anything in particular."
        })
    start: bool = field(
        default=False,
        metadata={
            "help":
            "Freeze the encoder layers and only train the layer between the encoder and classification architecture. Probably works best with --token flag since [CLS] may not be well-trained for anything in particular."
        })
    
    scale: float = field(
        default=30.0, metadata={"help": "scale value used for the arcface."})
    margin: float = field(
        default=0.5, metadata={"help": "margin value used for the arcface."})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CnlpTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        )

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # assert len(data_args.task_name) == len(
    #     data_args.data_dir
    # ), 'Number of tasks and data directories should be the same!'

    # baselines = ['cnn', 'lstm']

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s"
        % (training_args.local_rank, training_args.device, training_args.n_gpu,
           bool(training_args.local_rank != -1), training_args.fp16))
    logger.info("Training/evaluation parameters %s" % training_args)
    logger.info("Data parameters %s" % data_args)
    logger.info("Model paramters %s" % model_args)
    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = []
        output_mode = []
        tagger = []
        for task_name in data_args.task_name:
            num_labels.append(len(cnlp_processors[task_name]().get_labels()))
            output_mode.append(cnlp_output_modes[task_name])
            tagger.append(cnlp_output_modes[task_name] == tagging)

        # num_labels = [len(cnlp_processors[task_name]().get_labels() for task_name in data_args.task_name)
        # output_mode = [cnlp_output_modes[data_args.task_name]for task_name in data_args.task_name)
        # tagger = cnlp_output_modes[data_args.task_name] == tagging
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    model_name = model_args.model_name_or_path
    # if not model_name in ALL_PRETRAINED_CONFIG_ARCHIVE_MAP and not model_name in baselines:
    #     # we are loading one of our own trained models, load it as-is initially,
    #     # then delete its classifier head, save as temp file, and make that temp file
    #     # the model file to be loaded down below the normal way. since that temp file
    #     # doesn't have a stored classifier it will use the randomly-inited classifier head
    #     # with the size of the supplied config (for the new task)
    #     config = AutoConfig.from_pretrained(
    #         model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #         cache_dir=model_args.cache_dir,
    #     )
    #     model = CnlpRobertaForClassification.from_pretrained(
    #             model_args.model_name_or_path,
    #             config=config,
    #             cache_dir=model_args.cache_dir,
    #             layer=model_args.layer,
    #             tokens=model_args.token,
    #             freeze=model_args.freeze,
    #             tagger=tagger)
    #     delattr(model, 'classifier')
    #     tempmodel = tempfile.NamedTemporaryFile(dir=model_args.cache_dir)
    #     torch.save(model.state_dict(), tempmodel)
    #     model_name = tempmodel.name

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        num_labels_list=num_labels,
        finetuning_task=data_args.task_name,
    )
    if model_args.start ==True:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            add_prefix_space=True,
            use_fast=True,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
            additional_special_tokens=['<e>', '</e>'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            add_prefix_space=True,
            use_fast=True)
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
            # additional_special_tokens=['<e>', '</e>'])

    pretrained = True

    model = CnlpBertForConceptNorm(
        model_name,
        config=config,
        num_labels_list=num_labels,
        scale=model_args.scale,
        margin=model_args.margin,
        # cache_dir=model_args.cache_dir,
        layer=model_args.layer,
        tokens=model_args.token,
        freeze=model_args.freeze,
        tagger=tagger,
        concept_embeddings_pre=training_args.concept_embeddings_pre)
    if model_args.start ==True:
        model.bert_mention.resize_token_embeddings(len(tokenizer))

    train_batch_size = training_args.per_device_train_batch_size * max(
        1, training_args.n_gpu)

    # Get datasets
    train_dataset = (ClinicalNlpDataset(
        data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
                     if training_args.do_train else None)
    eval_dataset = (ClinicalNlpDataset(data_args,
                                       tokenizer=tokenizer,
                                       mode="dev",
                                       cache_dir=model_args.cache_dir)
                    if training_args.do_eval else None)
    test_dataset = (ClinicalNlpDataset(data_args,
                                       tokenizer=tokenizer,
                                       mode="test",
                                       cache_dir=model_args.cache_dir)
                    if training_args.do_predict else None)

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer,
                                                pad_to_multiple_of=8)
    else:
        data_collator = None

    best_eval_results = None
    output_eval_file = os.path.join(training_args.output_dir,
                                    f"eval_results.txt")
    output_test_file = os.path.join(training_args.output_dir,
                                    f"test_results.txt")
    # if training_args.do_train:
    #     batches_per_epoch = math.ceil(len(train_dataset) / train_batch_size)
    #     total_steps = int(training_args.num_train_epochs * batches_per_epoch //
    #                       training_args.gradient_accumulation_steps)

    # if training_args.evals_per_epoch > 0:
    #     logger.warning(
    #         'Overwriting the value of logging steps based on provided evals_per_epoch argument'
    #     )
    # steps per epoch factors in gradient accumulation steps (as compared to batches_per_epoch above which doesn't)
    # steps_per_epoch = int(total_steps // training_args.num_train_epochs)
    # training_args.eval_steps = steps_per_epoch // training_args.evals_per_epoch
    training_args.evaluation_strategy = IntervalStrategy.EPOCH
    training_args.save_strategy = IntervalStrategy.EPOCH
    training_args.logging_strategy = IntervalStrategy.STEPS
    training_args.logging_steps = 1000
    # training_args.eval_steps = 1
    # training_args.save_steps = 1

    # elif training_args.do_eval:
    #     logger.info(
    #         'Evaluation strategy not specified so evaluating every epoch')
    #     training_args.evaluation_strategy = EvaluationStrategy.EPOCH

    def build_compute_metrics_fn(task_names: List[str],
                                 model) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):

            metrics = {}
            task_scores = []
            # if not p is list:
            #     p = [p]

            for task_ind, task_name in enumerate(task_names):
                if tagger[task_ind]:
                    preds = np.argmax(p.predictions[task_ind], axis=2)
                    # labels will be -100 where we don't need to tag
                else:
                    if task_ind == 0:
                        preds = np.argmax(p.predictions[task_ind], axis=1)
                    else:
                        preds = p.predictions[task_ind][:, 0]

                if len(task_names) == 1:
                    labels = p.label_ids
                else:
                    labels = p.label_ids[task_ind]

                metrics[task_name] = cnlp_compute_metrics(
                    task_name, preds, labels)
                processor = cnlp_processors[task_name]()
                task_scores.append(processor.get_one_score(metrics[task_name]))

            one_score = sum(task_scores) / len(task_scores)

            if not model is None:
                if not hasattr(model,
                               'best_score') or one_score > model.best_score:
                    if pretrained:
                        trainer.save_model()
                    # For convenience, we also re-save the tokenizer to the same directory,
                    # so that you can share your model easily on huggingface.co/models =)
                    if trainer.is_world_process_zero():
                        tokenizer.save_pretrained(training_args.output_dir)
                        for task_ind, task_name in enumerate(metrics):
                            # with open(output_eval_file, "w") as writer:
                            logger.info(
                                "***** Eval results for task %s *****" %
                                (task_name))
                            for key, value in metrics[task_name].items():
                                if key == "eval_ner_test" and isinstance(
                                        value, dict):
                                    for key_key, key_value in value.items():
                                        logger.info("  %s = %s", key_key,
                                                    key_value)
                                        # writer.write("%s = %s\n" %
                                        #              (key_key, key_value))
                                else:
                                    logger.info("  %s = %s", key, value)
                                    # writer.write("%s = %s\n" %
                                    #              (key, value))
                    model.best_score = one_score
                    model.best_eval_results = metrics

            return metrics

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics_fn(data_args.task_name, model),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            # resume_from_checkpoint=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            resume_from_checkpoint=None)

        if not hasattr(model, 'best_score'):
            if pretrained:
                trainer.save_model()
                # For convenience, label_idswe also re-save the tokenizer to the same directory,
                # so that you can share your model easily on huggingface.co/models =)
                if trainer.is_world_process_zero():
                    tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        try:
            eval_result = model.best_eval_results
        except:
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in eval_result.items():
                    if key == "eval_ner_test" and isinstance(value, dict):
                        for key_key, key_value in value.items():
                            logger.info("  %s = %s", key_key, key_value)
                            writer.write("%s = %s\n" % (key_key, key_value))
                    else:
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

        eval_results.update(eval_result)
        
        # model..save_pretrained(save_path)

    if training_args.do_predict:
        if training_args.do_eval:
            predictions_tasks_eval, labels_tasks_eval, metrics_tasks_eval = trainer.predict(
                eval_dataset)

            for task_ind, task_name in enumerate(data_args.task_name):
                label_list_eval = cnlp_processors[task_name]().get_labels()

                predictions_task_id_eval = np.argsort(
                    predictions_tasks_eval[task_ind], axis=-1)
                predictions_task_id_eval = predictions_task_id_eval[:, ::
                                                                    -1][:, :30]

                score_array_eval = [
                    row[predictions_task_id_eval[i]]
                    for i, row in enumerate(predictions_tasks_eval[task_ind])
                ]

                np.save(
                    os.path.join(training_args.output_dir,
                                 "%s_eval_predictions" % task_name),
                    score_array_eval)

                true_predictions_eval = [[
                    label_list_eval[prediction] for prediction in predictions
                ] for predictions, label in zip(predictions_task_id_eval,
                                                labels_tasks_eval)
                                         if label != -100]

                # if task_ind == 0:
                #     predictions_task_id_eval = np.argsort(
                #         predictions_tasks_eval[task_ind], axis=-1)
                #     predictions_task_id_eval = predictions_task_id_eval[:, ::
                #                                                         -1][:, :
                #                                                             10]

                #     score_array_eval = [
                #         row[predictions_task_id_eval[i]] for i, row in
                #         enumerate(predictions_tasks_eval[task_ind])
                #     ]

                #     true_predictions_eval = [[
                #         label_list_eval[prediction]
                #         for prediction in predictions
                #     ] for predictions, label in zip(
                #         predictions_task_id_eval, labels_tasks_eval[task_ind])
                #                              if label != -100]

                #     np.save(
                #         os.path.join(training_args.output_dir,
                #                      "%s_eval_predictions" % task_name),
                #         score_array_eval)

                # else:

                #     true_predictions_eval = predictions_tasks_eval[task_ind]
                #     true_predictions_eval = [[
                #         label_list_eval[prediction]
                #         for prediction in predictions
                #     ] for predictions, label in zip(
                #         true_predictions_eval, labels_tasks_eval[task_ind])
                #                              if label != -100]

                output_eval_predictions_file = os.path.join(
                    training_args.output_dir,
                    "%s_eval_predictions.txt" % task_name)

                if trainer.is_world_process_zero():
                    with open(output_eval_predictions_file, "w") as writer:
                        for prediction in true_predictions_eval:
                            writer.write(" ".join(prediction) + "\n")

        ########################## Test Predictions #######################
        predictions_tasks, labels_tasks, metrics_tasks = trainer.predict(
            test_dataset)
        for task_ind, task_name in enumerate(data_args.task_name):

            label_list = cnlp_processors[task_name]().get_labels()
            predictions_tasks_id = np.argsort(predictions_tasks[task_ind],
                                              axis=-1)
            predictions_tasks_id = predictions_tasks_id[:, ::-1][:, :30]

            score_array = [
                row[predictions_tasks_id[i]]
                for i, row in enumerate(predictions_tasks[task_ind])
            ]

            output_test_predictions_file = os.path.join(
                training_args.output_dir,
                "%s_test_predictions.txt" % (task_name))
            np.save(
                os.path.join(training_args.output_dir,
                             "%s_test_predictions" % (task_name)), score_array)

            true_predictions = [[
                label_list[prediction] for prediction in predictions
            ] for predictions, label in zip(predictions_tasks_id, labels_tasks)
                                if label != -100]

            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results for task %s *****" %
                                (task_name))
                    for key, value in metrics_tasks.items():
                        if key == "eval_ner_test" and isinstance(value, dict):
                            for key_key, key_value in value.items():
                                logger.info("  %s = %s", key_key, key_value)
                                writer.write("%s = %s\n" %
                                             (key_key, key_value))
                        else:
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))

            # Save predictions
            output_test_predictions_file = os.path.join(
                training_args.output_dir,
                "%s_test_predictions.txt" % (task_name))

            if trainer.is_world_process_zero():
                with open(output_test_predictions_file, "w") as writer:
                    for prediction in true_predictions:
                        writer.write(" ".join(prediction) + "\n")

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
