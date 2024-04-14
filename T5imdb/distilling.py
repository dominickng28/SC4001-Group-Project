import accelerate
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from datasets import Dataset, DatasetDict
import datasets
import numpy as np
import random
from sklearn.model_selection import train_test_split
import wandb
import argparse
import json
# Arg Parser
parser = argparse.ArgumentParser()
parser.add_argument('--subsample', type=float, default=1.0)
parser.add_argument('--pretrained', type=str, default='google/t5-v1_1-small')
args = parser.parse_args()

# For the json dataset
data_files = {
            'train': "./T5imdb/imdb/llm/imdb_llm_train.json",
            'test': "./T5imdb/imdb/llm/imdb_llm_test.json",
        }

json_dataset = load_dataset("json", data_files=data_files)

# Subsample json dataset
dataset_size = args.subsample
if (0 < dataset_size < 1):
    _, train_dataset= json_dataset['train'].train_test_split(test_size=dataset_size, seed=42).values()
    json_dataset['train'] = train_dataset

# json train val split
train_dataset, val_dataset = json_dataset['train'].train_test_split(test_size=0.3, seed=42).values()
json_dataset['train'] = train_dataset
json_dataset['val'] = val_dataset


#define model, tokenizer, data collator
model = T5ForConditionalGeneration.from_pretrained(args.pretrained)
tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Compute metrics distilling
def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics

compute_metrics = compute_metrics_text(tokenizer)

# tokenize function distilling
def tokenize_func(example):
    # Tokenize text
    tokenized_inputs = tokenizer(example["question"], max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
      label_output_encodings = tokenizer(example["answer"], max_length=512, truncation=True)

    tokenized_inputs['labels'] = label_output_encodings['input_ids']
    # Return tokenized inputs and decoder input IDs
    return tokenized_inputs

# Tokenize the dataset
tokenized_dataset = json_dataset.map(tokenize_func, remove_columns=['id', 'question', 'choices', 'answer', 'rationale'], batched=True)


training_args = Seq2SeqTrainingArguments(
    output_dir = '/home/UG/ng0003ck/T5imdb',
    remove_unused_columns = False,
    evaluation_strategy = 'steps',
    save_strategy='no',
    logging_dir= "/home/UG/ng0003ck/T5imdb/logs",
    logging_steps= 100,
    max_steps=3500,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    seed=42,
    generation_max_length=64,
    run_name="T5-Distilling",
    prediction_loss_only=False,
)



trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['val'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
    )



trainer.train()


predict = trainer.predict(test_dataset = tokenized_dataset['test'])
print(predict)
predicted_texts = []
actual_texts = []

for i in range(len(predict.predictions)):
    pred_text = tokenizer.decode(predict.predictions[i], skip_special_tokens=True)
    predicted_texts.append(pred_text)
actual_texts = json_dataset['test']['answer']

# To view the first 100 results
print(predicted_texts[:100])
print(actual_texts[:100])

print("test Accuracy:")
print(np.mean(np.array(predicted_texts) == np.array(actual_texts)))

