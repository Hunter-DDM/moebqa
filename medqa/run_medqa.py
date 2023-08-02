#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import datasets
import numpy as np
import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from transformers.models.bert.configuration_bert import BertConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.models.bert.modeling_bert import BertForMultipleChoice
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from custom_moe_bert import MoEBertForMultipleChoice

import re

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.13.0.dev0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    moe_bert: bool = field(
        default=False,
        metadata={"help": "whether to use the moe version of bert"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    routing_feat: Optional[str] = field(
        default='cls', metadata={"help": "What feature is used for routing"}
    )
    gate_type: Optional[str] = field(
        default='v1', metadata={"help": "What gate type to use"}
    )
    balance_type: Optional[str] = field(
        default='acc_all', metadata={"help": "What balance controling method to use"}
    )
    balance_factor: Optional[float] = field(
        default=0.01, metadata={"help": "Balance factor used for computing balance loss"}
    )
    moe_arch: Optional[str] = field(
        default='4-4-4', metadata={"help": "Layers for the bottom, middle MoE, and top modules, connected by '-'"}
    )
    num_expert: Optional[int] = field(
        default=4, metadata={"help": "Number of experts in for MoE"}
    )
    uniform_ln: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a uniform LayerNorm after the MoE layer"}
    )
    routing_linear: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a Linear Layer to process the routing feature"}
    )
    routing_dim: Optional[int] = field(
        default=20, metadata={"help": "The dimension of routing feature, valid only if routing_linear is set to True"}
    )
    random_pooler_dense: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a randomly initialized dense layer in the pooler layer"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the accuracy"},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the accuracy"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of testing examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.train_file is not None or data_args.validation_file is not None or data_args.test_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, field='data', data_files=data_files, cache_dir=model_args.cache_dir)
    else:
        raise ValueError('Need a data file (train, dev, or test)')

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = BertConfig.from_pretrained(
        model_args.model_name_or_path,
    )
    tokenizer = BertTokenizerFast.from_pretrained(
        model_args.model_name_or_path,
    )
    if model_args.moe_bert:
        config.routing_feat = model_args.routing_feat
        config.gate_type = model_args.gate_type
        config.balance_type = model_args.balance_type
        config.balance_factor = model_args.balance_factor
        config.moe_arch = model_args.moe_arch
        config.num_expert = model_args.num_expert
        config.uniform_ln = model_args.uniform_ln
        config.routing_linear = model_args.routing_linear
        config.routing_dim = model_args.routing_dim
        config.random_pooler_dense = model_args.random_pooler_dense
        model = MoEBertForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        if training_args.do_train:
            model.bert.encoder.init_moe_module()
        else:
            del model.bert.encoder.layer
    else:
        model = BertForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )


    # # ! used for analysis
    # def flatten_params(modules):
    #     parameter_list = []
    #     for param in modules.parameters():
    #         parameter_list.append(param.view(-1))
    #     params = torch.cat(parameter_list, dim=0)
    #     print(params.shape)
    #     return params


    # print('=============' * 10)
    # flattened_params = []
    # for i in range(4):
    #     flattened_params.append(flatten_params(model.bert.encoder.middle_moe_layer[i]))
    # flattened_params.append(flatten_params(model.bert.encoder.middle_shared_layer))
    # similarity = torch.zeros(5, 5)
    # cos_sim = torch.nn.CosineSimilarity(dim=0)
    # for i in range(5):
    #     for j in range(5):
    #         similarity[i, j] = cos_sim(flattened_params[i], flattened_params[j])
    #         # similarity[i, j] = torch.sum(flattened_params[i] * flattened_params[j])
    # print(similarity.tolist())

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    question_name = "qtext"
    answer_name = "answers"

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Preprocessing the datasets.
    def preprocess_function(examples):
        first_sentences = [[context.strip()] * 4 for context in examples[question_name]]
        answers = examples[answer_name]
        second_sentences = [
            [answer[i].strip() for i in range(len(answer))] for answer in answers
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation="only_first",
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
            return_offsets_mapping=True,
        )

        question_char_pos = []
        for i, first_sentence in enumerate(first_sentences):
            question_text = re.split(r'[?.\n]', first_sentence[:-5])[-1] + first_sentence[-5:]
            question_text = question_text.strip()
            question_stt = first_sentence.find(question_text)
            question_end = question_stt + len(question_text)
            question_char_pos.append([question_stt, question_end])

        offsets = tokenized_examples.pop("offset_mapping")
        tokenized_examples['q_stt'] = []
        tokenized_examples['q_end'] = []

        for i, offset in enumerate(offsets):
            q_stt = 0
            while tokenized_examples.sequence_ids(i)[q_stt] != 0:
                q_stt += 1
            while q_stt < len(offset) and offset[q_stt][0] < question_char_pos[i][0]:
                q_stt += 1
            q_end = len(offset) - 1
            while tokenized_examples.sequence_ids(i)[q_end] != 0:
                q_end -= 1
            while q_end > 0 and offset[q_end][1] > question_char_pos[i][1]:
                q_end -= 1
            q_end += 1
            tokenized_examples['q_stt'].append(q_stt)
            tokenized_examples['q_end'].append(q_end)

        # Un-flatten
        rlts = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        labels = {'labels': examples['labels']}
        rlts = {**rlts, **labels}
        return rlts

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="test dataset map pre-processing"):
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    # Data collator
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if training_args.do_predict:
        logger.info("*** Test ***")

        results = trainer.predict(test_dataset)
        metrics = results.metrics
        max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

if __name__ == "__main__":
    main()