import logging
import os
import random
import sys
import math
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator

from model_args import ModelArguments
from data_args import DataTrainingArguments
import constants

import inspect
import datasets
import numpy as np
from datasets import load_dataset, load_metric
import torch
import json

import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
    set_seed,
    Trainer
)
# from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_glue import LGTMTeacher,cal_loss

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)


def _remove_unused_columns(model, dataset: "datasets.Dataset", description: Optional[str] = None):
    # if not self.args.remove_unused_columns:
    #     return dataset
    # if _signature_columns is None:
    # Inspect model forward signature to keep only the arguments it accepts.
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
    
def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator()
    os.makedirs(training_args.output_dir, exist_ok=True)

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
    
    # Labels
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
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

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
    
    # teacher model(only used in training)
    if training_args.do_train:
        t_config = AutoConfig.from_pretrained(model_args.teacher_model, num_labels=num_labels, finetuning_task=data_args.task_name)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.teacher_model,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=t_config,
        )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = constants.task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        raw_datasets = _remove_unused_columns(model, raw_datasets)
        
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
            
    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # prediction
    if training_args.do_predict:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            
            output_predict_file = os.path.join(training_args.output_dir, f"{task}.tsv")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
        return

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    if model_args.train_teacher:
        t_optimizer = AdamW(teacher_model.parameters(), lr=model_args.t_learning_rate)
    
        if model_args.init_classifier_to_zero:
            init_classifier_as_zero(teacher_model)
            init_classifier_as_zero(model)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    num_warmup_steps = max_train_steps*training_args.warmup_ratio

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    if model_args.train_teacher:
        t_lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=t_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        
        model, teacher_model, optimizer, t_optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, teacher_model, optimizer, t_optimizer, train_dataloader, eval_dataloader
        )

        if model_args.use_lgtm:
            held_iter = iter(eval_dataloader)
            model_total = LGTMTeacher(teacher_model, model, model_args.alpha_kd, model_args.t_alpha_kd,
                                    t_optimizer, t_lr_scheduler, model_args.temperature)
            
    else:
        model, teacher_model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
                model, teacher_model, optimizer, train_dataloader, eval_dataloader
            )
     
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {int(training_args.num_train_epochs)}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0
    best_metric = 0.0
    t_best_metric = 0.0

    for epoch in range(int(training_args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            # use teacher logits as soft labels
            teacher_model.eval()
            model.train()
                        
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                t_logits = teacher_outputs.logits

            outputs = model(**batch)
            loss, logits = outputs.loss, outputs.logits
            loss = model_args.alpha_kd * cal_loss(logits, t_logits, model_args.temperature) + (1-model_args.alpha_kd) * loss
        
            # update the student
            loss.backward()

            model.eval()
            teacher_model.train()

            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
            teacher_outputs = teacher_model(**batch)
            t_loss, t_logits = teacher_outputs.loss, teacher_outputs.logits
            t_loss = model_args.t_alpha_kd * cal_loss(t_logits,logits, model_args.temperature) + (1 - model_args.t_alpha_kd) * t_loss
            
            # update the teacher
            t_loss.backward()
        
            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                t_optimizer.step()
                t_lr_scheduler.step()
                t_optimizer.zero_grad()

            # update the student 
            loss = loss / int(training_args.gradient_accumulation_steps)
            loss.backward()

            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            # We keep track of the loss at each epoch
            if completed_steps % training_args.eval_steps == 0 or completed_steps == max_train_steps:
                model.eval()
                samples_seen = 0

                # student evaluation
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                    predictions, references = accelerator.gather((predictions, batch["labels"]))
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(eval_dataloader) - 1:
                            predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                            references = references[: len(eval_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += references.shape[0]
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )           
                    # eval_metric = metric.compute(predictions=predictions, references=batch["labels"])
                    # acc_metric += eval_metric['accuracy']
                    # f1_metric += eval_metric['f1']
                eval_metric = metric.compute()
                
                logger.info("***** Evaluation Results*****")
                logger.info(f"  Training step = {completed_steps}")
                for key, value in eval_metric.items():
                    logger.info(f" eval_{key}:{value} ")
           
                if data_args.task_name in constants.acc_tasks:
                    metric_key = "accuracy"
                elif data_args.task_name in constants.f1_tasks:
                    metric_key = "f1"
                
                if eval_metric[metric_key] > best_metric:
                    best_metric = eval_metric[metric_key]
                    tokenizer.save_pretrained(training_args.output_dir)
                    model.save_pretrained(training_args.output_dir)
                    path = os.path.join(training_args.output_dir, "eval_results.json")
                    with open(path, "w") as f:
                        json.dump(eval_metric, f, indent=4, sort_keys=True)

                # teacher evaluation 
                teacher_model.eval()
                samples_seen = 0
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = teacher_model(**batch)
                    predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                    predictions, references = accelerator.gather((predictions, batch["labels"]))
                    if accelerator.num_processes > 1:
                        if step == len(eval_dataloader) - 1:
                            predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                            references = references[: len(eval_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += references.shape[0]
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )

                eval_metric = metric.compute()
                logger.info("***** Teacher Evaluation Results*****")
                logger.info(f"  Training step = {completed_steps}")
                for key, value in eval_metric.items():
                    logger.info(f" eval_{key}:{value} ")  
                
                if eval_metric[metric_key] > t_best_metric:
                    t_best_metric = eval_metric[metric_key]
                    path = os.path.join(training_args.output_dir, "teacher_eval_results.json")
                    with open(path, "w") as f:
                        json.dump(eval_metric, f, indent=4, sort_keys=True) 

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
