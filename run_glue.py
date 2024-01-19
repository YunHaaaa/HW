import logging
import os
import random
import sys
import math
from accelerate import Accelerator

from pruning.utils import get_weight_threshold, weight_prune, get_filter_mask, filter_prune, cal_sparsity
from model_args import ModelArguments
from data_args import DataTrainingArguments
import constants
import utils

import numpy as np
from datasets import load_metric
import torch
import json

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
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
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_glue import LGTMTeacher,cal_loss

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)

    
def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator()
    os.makedirs(training_args.output_dir, exist_ok=True)
    utils.setup_logging(training_args)

    set_seed(training_args.seed)
    
    raw_datasets = utils.load_glue_dataset(data_args, model_args, training_args)
    is_regression, num_labels, label_list = utils.process_labels(data_args, raw_datasets)
    config, model = utils.load_model(model_args, data_args, num_labels)
    tokenizer = utils.load_tokenizer(model_args)

    # teacher model(only used in training)
    if training_args.do_train:
        # t_config, teacher_model = utils.load_model()
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
        raw_datasets = utils.remove_unused_columns(model, raw_datasets)
        
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
            output_predict_file = os.path.join(training_args.output_dir, f"{task}.tsv")
            utils.perform_prediction(trainer, predict_dataset, task, output_predict_file, is_regression, label_list)

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
            utils.init_classifier_as_zero(teacher_model)
            utils.init_classifier_as_zero(model)

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
    
    # TODO: Teacher, Student predict -> check performance

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

    # One-shot magnitude pruning for teacher model
    if training_args.do_train:
        threshold = get_weight_threshold(teacher_model, rate=0.8, args=model_args)
        weight_prune(teacher_model, threshold, model_args)

    # Print teacher model state_dict after pruning
    print("Teacher Model State_dict After Pruning:")
    for name, param in teacher_model.state_dict().items():
        print(name, param)

    for epoch in range(int(training_args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):

            # TODO: 이대로 테스트 해보고 합쳐서 확인
            # use teacher logits as soft labels
            teacher_model.eval()
            model.train()
                        
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                t_logits = teacher_outputs.logits

            outputs = model(**batch)
            loss, logits = outputs.loss, outputs.logits
            loss = model_args.alpha_kd * cal_loss(logits, t_logits, model_args.temperature) + (1 - model_args.alpha_kd) * loss
        
            # update the student
            loss.backward()

            model.eval()
            teacher_model.train()

            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits

            teacher_outputs = teacher_model(**batch)
            t_loss, t_logits = teacher_outputs.loss, teacher_outputs.logits
            t_loss = model_args.t_alpha_kd * cal_loss(t_logits, logits, model_args.temperature) + (1 - model_args.t_alpha_kd) * t_loss
            
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
