import pathlib
import argparse
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM
from transformers.trainer_utils import get_last_checkpoint
import wandb

from .. import Sample

os.environ["WANDB_PROJECT"] = "coq_learn.tactic_predict"
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"

parser = argparse.ArgumentParser()
parser.add_argument("--train-dataset-file", help="Train dataset location", type=pathlib.Path)
parser.add_argument("--eval-dataset-file", help="Evaluation dataset location", type=pathlib.Path)
parser.add_argument(
    "--output-dir",
    help="Directory where the tokenizer, model parameters and logs get stored",
    type=pathlib.Path,
)
parser.add_argument("--train-epochs", help="How many passes over the training data", type=int)
parser.add_argument(
    "--device-batch-size",
    help="How many samples per device per batch (lower if memory issues)",
    type=int,
)
args = parser.parse_args()

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

dataset = load_dataset(
    "text",
    data_files={
        "train": str(args.train_dataset_file),
        "eval": str(args.eval_dataset_file),
    },
)


def preprocess_function(examples):
    """split goal/tactic, tokenize the text, and set the labels"""
    samples = [Sample.model_validate_json(line) for line in examples["text"]]
    (goals, tactics) = [sample.goals for sample in samples], [sample.tactic for sample in samples]

    model_inputs = tokenizer(goals, max_length=128, truncation=True)

    # The "labels" are the tokenized outputs:
    labels = tokenizer(text_target=tactics, max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    # return {"inputs_id": model_inputs["input_ids"], "labels": labels["input_ids"]}


tokenized_dataset = dataset.map(preprocess_function, batched=True)

if os.path.isdir(args.output_dir):
    print("Trying to resume from a checkpoint")
    resume_from_checkpoint = True
else:
    resume_from_checkpoint = False

training_args = Seq2SeqTrainingArguments(
    args.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_safetensors=False,
    logging_strategy="steps",
    logging_steps=100,
    learning_rate=4e-5,
    per_device_train_batch_size=args.device_batch_size,
    per_device_eval_batch_size=args.device_batch_size,
    weight_decay=0.01,
    num_train_epochs=args.train_epochs,
    predict_with_generate=True,
    fp16=False,
    load_best_model_at_end=False,
    report_to="wandb",
)

data_collator = DataCollatorForSeq2Seq(tokenizer)


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    data = {"inputs": dataset["eval"]["text"], "labels": decoded_labels, "preds": decoded_preds}
    wandb.log({"eval": wandb.Table(dataframe=pd.DataFrame.from_dict(data))})

    return {"sentence_accuracy": sum([decoded_pred == decoded_label for decoded_pred, decoded_label in zip(decoded_preds, decoded_labels)]) / len(labels)}


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=resume_from_checkpoint)
