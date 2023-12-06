import os
import logging
import sys
import datasets
import inspect
from datasets import load_dataset
import numpy as np
import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from typing import Optional

logger = logging.getLogger(__name__)

def remove_unused_columns(model, dataset: "datasets.Dataset", description: Optional[str] = None):
    signature = inspect.signature(model.forward)
    _signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    _signature_columns += ["label", "label_ids"]
    columns = [k for k in _signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset['train'].column_names) - set(_signature_columns))
    if len(ignored_columns) > 0:
        dset_description = "" if description is None else f"in the {description} set "
        logger.info(
            f"The following columns {dset_description} don't have a corresponding argument in "
            f"`{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
        )
    return dataset.remove_columns(ignored_columns)

def init_classifier_as_zero(model):
    for params in model.classifier.parameters():
        params.data.fill_(0.0)

def setup_logging(training_args):
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

    # 함수 호출
    setup_logging(training_args)


def load_glue_dataset(data_args, model_args, training_args):
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        # Loading a dataset from your local files.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)

    return raw_datasets


def process_labels(data_args, raw_datasets):
    is_regression = False
    num_labels = 0
    label_list = None

    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    return is_regression, num_labels, label_list


def load_model_and_tokenizer(model_args, data_args, num_labels):
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        num_hidden_layers=model_args.num_layers
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    return config, tokenizer, model


def perform_prediction(trainer, predict_dataset, task, output_predict_file, is_regression=False, label_list=None):
    predict_dataset = predict_dataset.remove_columns("label")
    predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
    predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

    with open(output_predict_file, "w") as writer:
        logger.info(f"***** Predict results {task} *****")
        writer.write("index\tprediction\n")
        for index, item in enumerate(predictions):
            if is_regression:
                writer.write(f"{index}\t{item:3.3f}\n")
            else:
                item = label_list[item]
                writer.write(f"{index}\t{item}\n")
