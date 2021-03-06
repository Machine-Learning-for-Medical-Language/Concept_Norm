import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from os.path import basename, dirname
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import DataProcessor, InputExample
from transformers.tokenization_utils import PreTrainedTokenizer

from cnlp_processors import cnlp_output_modes, cnlp_processors

logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    input_ids_c: List[int] = None
    attention_mask_c: Optional[List[int]] = None
    token_type_ids_c: Optional[List[int]] = None
    event_tokens: Optional[List[int]] = None
    event_tokens_c: Optional[List[int]] = None
    concept_labels: List[Optional[Union[int, float, List[int]]]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def cnlp_convert_examples_to_features(examples: List[InputExample],
                                      tokenizer: PreTrainedTokenizer,
                                      task=None,
                                      label_lists=None,
                                      output_mode=None,
                                      token_classify=False,
                                      pad_to_max_length=False,
                                      max_seq_length=None):
    event_start_ind = tokenizer.convert_tokens_to_ids('<e>')
    event_end_ind = tokenizer.convert_tokens_to_ids('</e>')

    # if task is not None:
    #     processor = cnlp_processors[task]()
    #     if label_lists is None:
    #         st_labels, concept_labels = processor.get_labels()
    #         logger.info("Using label list %s for task %s" % (st_labels, task))
    #     if output_mode is None:
    #         output_mode = cnlp_output_modes[task]
    #         logger.info("Using output mode %s for task %s" %
    #                     (output_mode, task))

    concept_label_map = {label: i for i, label in enumerate(label_lists[0])}

    def label_from_example(example: InputExample, label_map: dict,
                           idx: int) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            try:
                return label_map[example.label.split("+++")[idx]]
            except:
                logger.error('Error with example %s' % (example.guid))
                raise Exception()

        elif output_mode == "regression":
            return float(example.label)
        elif output_mode == 'tagging':
            return [label_map[label] for label in example.label]

        raise KeyError(output_mode)

    if "concept_normalization" in task:
        labels_concept = [
            label_from_example(example, concept_label_map, 1)
            for example in examples
        ]
    else:
        labels_concept = [
            label_from_example(example, concept_label_map, 0)
            for example in examples
        ]

    sentences = [example.text_a for example in examples]

    if max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    padding = "max_length" if pad_to_max_length else False

    batch_encoding = tokenizer(
        sentences,
        padding=padding,
        max_length=max_seq_length,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
    )

    # This code has to solve the problem of properly setting labels for word pieces that do not actually need to be tagged.

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        try:
            event_start = inputs['input_ids'].index(event_start_ind)
        except:
            event_start = -1

        try:
            event_end = inputs['input_ids'].index(event_end_ind)
        except:
            event_end = len(inputs['input_ids']) - 1

        inputs['event_tokens'] = [0] * len(inputs['input_ids'])
        if event_start >= 0:
            inputs['event_tokens'] = [0] * event_start + [1] * (
                event_end - event_start +
                1) + [0] * (len(inputs['input_ids']) - event_end - 1)
        else:
            inputs['event_tokens'] = [1] * len(inputs['input_ids'])

        if labels_concept[0] is not None:

            feature = InputFeatures(
                **inputs,
                concept_labels=labels_concept[i],
            )
        else:
            feature = InputFeatures(**inputs)

        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: List[str] = field(
        metadata={
            "help":
            "The input data dirs. A space-separated list of directories that should contain the .tsv files (or other data files) for the task. Should be presented in the same order as the task names."
        })

    task_name: List[str] = field(
        default_factory=lambda: None,
        metadata={
            "help":
            "A space-separated list of tasks to train on: " +
            ", ".join(cnlp_processors.keys())
        })
    # field(

    #     metadata={"help": "A space-separated list of tasks to train on: " + ", ".join(cnlp_processors.keys())})

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help":
            "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"})


class ClinicalNlpDataset(Dataset):
    """ Copy-pasted from GlueDataset with glue task-specific code changed
        moved into here to be self-contained
    """
    args: DataTrainingArguments
    output_mode: List[str]
    features: List[InputFeatures]

    def __init__(
        self,
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processors = [cnlp_processors[task]() for task in args.task_name]
        self.output_mode = [cnlp_output_modes[task] for task in args.task_name]
        self.features = None

        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        # Load data features from cache or dataset file
        self.label_lists = [
            processor.get_labels() for processor in self.processors
        ]

        for task_ind, data_dir in enumerate(args.data_dir):
            datadir = dirname(data_dir) if data_dir[-1] == '/' else data_dir
            domain = basename(datadir)
            dataconfig = basename(dirname(datadir))

            cached_features_file = os.path.join(
                cache_dir if cache_dir is not None else data_dir,
                "cached_{}-{}_{}_{}".format(dataconfig, domain, mode.value,
                                            tokenizer.__class__.__name__),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(
                        cached_features_file) and not args.overwrite_cache:
                    start = time.time()
                    features = torch.load(cached_features_file)
                    logger.info(
                        f"Loading features from cached file {cached_features_file} [took %.3f s]",
                        time.time() - start)
                else:
                    logger.info(
                        f"Creating features from dataset file at {data_dir}")

                    if mode == Split.dev:
                        examples = self.processors[task_ind].get_dev_examples(
                            data_dir)
                    elif mode == Split.test:
                        examples = self.processors[task_ind].get_test_examples(
                            data_dir)
                    else:
                        examples = self.processors[
                            task_ind].get_train_examples(data_dir)
                    if limit_length is not None:
                        examples = examples[:limit_length]
                    features = cnlp_convert_examples_to_features(
                        examples,
                        tokenizer,
                        task=args.task_name,
                        label_lists=self.label_lists,
                        output_mode=self.output_mode[task_ind],
                        pad_to_max_length=args.pad_to_max_length,
                        max_seq_length=args.max_seq_length)
                    start = time.time()
                    torch.save(features, cached_features_file)
                    # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                    logger.info(
                        "Saving features into cached file %s [took %.3f s]",
                        cached_features_file,
                        time.time() - start)

                if self.features is None:
                    self.features = features
                else:
                    # we should have all non-label features be the same, so we can essentially discard subsequent
                    # datasets input features. So we'll append the labels from that features list and discard the duplicate
                    # input features.
                    assert len(features) == len(self.features)
                    for feature_ind, feature in enumerate(features):
                        self.features[feature_ind].label.append(
                            feature.label[0])

                    # self.features.label.append(features.labels[0])

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i].__dict__

    def get_labels(self):
        return self.label_lists
