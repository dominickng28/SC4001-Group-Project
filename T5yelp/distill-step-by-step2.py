import accelerate
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2Tokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed
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
            'train': "/home/UG/ng0003ck/T5yelp/yelp/llm/yelp_llm_train.json",
            'test': "/home/UG/ng0003ck/T5yelp/yelp/llm/yelp_llm_test.json",
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

#define model, tokenizer
model = T5ForConditionalGeneration.from_pretrained(args.pretrained)
tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

# Compute metrics distill-step-by-step
def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        preds = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics

compute_metrics = compute_metrics_text(tokenizer)

# tokenize function distill-step-by-step
def tokenize_func(examples):
    model_inputs = tokenizer(['predict: ' + text for text in examples['question']], max_length=512, truncation=True)
    expl_model_inputs = tokenizer(['explain: ' + text for text in examples['question']], max_length=512, truncation=True)
    model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
    model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

    with tokenizer.as_target_tokenizer():
        label_output_encodings = tokenizer(examples['answer'], max_length=512, truncation=True)
        rationale_output_encodings = tokenizer(examples['rationale'], max_length=512, truncation=True)

    model_inputs['labels'] = label_output_encodings['input_ids']
    model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

    return model_inputs

tokenized_dataset = json_dataset.map(tokenize_func, remove_columns=['id', 'question', 'choices', 'answer', 'rationale'], batched=True)

'''
TaskprefixDataCollator and Trainer referenced from:
Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes (Hsieh et al., 2023)
'''

# distill step-by-step data collator
class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids', 'expl_attention_mask': 'attention_mask'}).to_dict('records')
        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)
        return {
            'pred': pred_features,
            'expl': expl_features,
        }

data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)

# distill step-by-step trainer alpha fixed at 0.5, output rationale true
class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = 0.5
        self.output_rationale = True


    def compute_loss(self, model, inputs, return_outputs=False):
        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])

        loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss

        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        if self.output_rationale:
            expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        else:
            expl_outputs = pred_outputs # placeholder only

        loss = self.alpha * pred_outputs[0]  + (1 - self.alpha) * expl_outputs[0]

        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )

training_args = Seq2SeqTrainingArguments(
    output_dir = '/home/UG/ng0003ck/T5yelp',
    remove_unused_columns = False,
    evaluation_strategy = 'steps',
    save_strategy='no',
    logging_dir= "/home/UG/ng0003ck/T5yelp/logs",
    logging_steps= 100,
    max_steps=3500,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    seed=42,
    generation_max_length=64,
    run_name="T5-Distill-Step-By-Step",
    prediction_loss_only=False,
)



trainer = TaskPrefixTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['val'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)



trainer.train()


predict = trainer.predict(test_dataset = tokenized_dataset['test'])
print(predict)
predicted_texts = []
actual_texts = []
print(predict.predictions)
for i in range(len(predict.predictions[0])):
    pred_text = tokenizer.decode(predict.predictions[0][i], skip_special_tokens=True)
    predicted_texts.append(pred_text)
actual_texts = json_dataset['test']['answer']

# To view the first 100 results
print(predicted_texts[:100])
print(actual_texts[:100])

print("test Accuracy:")
print(np.mean(np.array(predicted_texts) == np.array(actual_texts)))

