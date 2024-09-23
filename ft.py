import torch
import torch.nn as nn
import pandas as pd
import numpy as np; import random
import os
from sadice import SelfAdjDiceLoss
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union, List, Dict
from transformers.data.data_collator import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from transformers import Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

import argparse
import wandb

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False) 


# run arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1, help="manual random seed")
parser.add_argument("--bs", type=int, default=512, help="")
parser.add_argument("--ep", type=int, default=10, help="")
parser.add_argument("--warmup_step", type=int, default=50, help="")
parser.add_argument("--lr", type=float, default=5e-5, help="")
parser.add_argument("--wd", type=float, default=1e-3, help="")
parser.add_argument("--drop_p", type=float, default=0.1, help="")
parser.add_argument("--losstype", type=str, default='ce', help="ce / gaff / dice / focal")

parser.add_argument("--use_pooled", action='store_true', help="pooled emb vs [CLS] emb")

parser.add_argument("--focal_gamma", type=float, default=1.0, help="")
parser.add_argument("--gaff_sigma", type=float, default=1.0, help="")
parser.add_argument("--gaff_margin", type=float, default=1.0, help="")

parser.add_argument("--mmixup", type=int, default=0, help="0 / 1 manifold mixup flag")
parser.add_argument("--betaparam", type=float, default=0.4, help="mmixup beta dist param")

parser.add_argument("--report_name", type=str, default='test', help="")
parser.add_argument("--wb_name", type=str, default='0919techbert', help="")

args = parser.parse_args()


# path 
vocab_path: str = './vocab/'
model_path: str = './pretrained/40ep_train9/'
data_path: str = './data/total_v2.1.csv'
test_path: str = './data/etri_changdae_AFTER.csv'
output_path: str = f'./finetune_output/{args.report_name}'

if os.path.isdir(output_path): pass
else: os.mkdir(output_path)

wandb.init(project=args.wb_name)
wandb.config.update(args)
wandb.run.name = output_path

# set seed for reproduction
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
set_seed(args.seed)

# data prepare
df = pd.read_csv(data_path)
df = df.dropna()

# remove etri-source train data
s = df.source=='etri'
df = df[~s].reset_index(drop=True)
test_df = pd.read_csv(test_path, encoding='cp949')
print(test_df.shape)
print(test_df.isna().sum())
test_df.tag = test_df.tag.replace({'기술의필요성':'기술의 필요성'})
test_df.tag = test_df.tag.replace({'기술동향':'기술의 파급효과'})

# build label dictionary
possible_labels = df.tag.unique()
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

# label encoding
df['labels'] = df.tag.replace(label_dict)

test_df = test_df[test_df.tag.apply(lambda x: x in label_dict.keys())]
test_df['labels'] = test_df.tag.replace(label_dict)


set_seed(args.seed)
# train : valid = 70 : 30 /// test : etri dataset
X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.labels.values, 
                                                  test_size=0.30, random_state=args.seed, stratify=df.labels.values)
X_val, X_test, y_val, y_test = train_test_split(df.loc[X_val].index.values, df.loc[X_val].labels.values, 
                                                  test_size=0.66, random_state=args.seed, stratify=df.loc[X_val].labels.values)

train, val, test = df.loc[X_train, ['text', 'labels']].reset_index(drop=True), df.loc[X_val, ['text', 'labels']].reset_index(drop=True), df.loc[X_test, ['text', 'labels']].reset_index(drop=True)

etri_test, etri_train= train_test_split(test_df.doc_id.unique(), test_size=0.30, random_state=args.seed)
etri_train, etri_valid= train_test_split(etri_train, test_size=0.33, random_state=args.seed)
e_train, e_valid, e_test = test_df.loc[test_df.doc_id.apply(lambda x: x in etri_train), ['text', 'labels']], test_df.loc[test_df.doc_id.apply(lambda x: x in etri_valid), ['text', 'labels']], test_df.loc[test_df.doc_id.apply(lambda x: x in etri_test),['text', 'labels']]

train, val, test = pd.concat([train, e_train], axis=0), pd.concat([val, e_valid], axis=0), pd.concat([test, e_test], axis=0)

# make Dataset
train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)
test_dataset = Dataset.from_pandas(test)
dataset = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})

tokenizer = BertTokenizerFast.from_pretrained(vocab_path, do_lower_case=False, model_max_length=128)

def transform(sentences: Union[str, List[str]], tokenizer) -> Dict[str, List[List[int]]]:
    if isinstance(sentences, str):
        sentences = [sentences]
    return tokenizer(text=sentences, add_special_tokens=True, padding=False, truncation=True, max_length = 128) ## Modify

train_ds = train_dataset.map(lambda data: transform(data["text"], tokenizer), remove_columns=["text"], batched=True)
valid_ds = val_dataset.map(lambda data: transform(data["text"], tokenizer), remove_columns=["text"], batched=True)
test_ds = test_dataset.map(lambda data: transform(data["text"], tokenizer), remove_columns=["text"], batched=True)

# BERT Pre-trained Model
model = BertForSequenceClassification.from_pretrained(model_path,
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False,
                                                      classifier_dropout = args.drop_p) #! need check

def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1_micro = f1_score(y_true=labels, y_pred=pred, average='micro')
    f1_macro = f1_score(y_true=labels, y_pred=pred, average='macro')
    recall = recall_score(y_true=labels, y_pred=pred, average='micro')
    precision = precision_score(y_true=labels, y_pred=pred, average='micro')
    
    return {"accuracy": accuracy, "f1_micro": f1_micro, 'f1_macro': f1_macro, 'recall': recall, 'precision': precision}

batchify = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
)

training_args = TrainingArguments(
    output_dir=output_path,          
    evaluation_strategy="epoch",
    eval_steps=1000,
    per_device_train_batch_size=args.bs,           #! 
    per_device_eval_batch_size=args.bs,            #! 
    learning_rate=args.lr,                         #! 
    weight_decay=args.wd,                          #! 
    adam_beta1=.9,
    adam_beta2=.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.,
    optim = 'adamw_hf',
    num_train_epochs=args.ep,                      #! 
    lr_scheduler_type="linear",
    warmup_steps=40,
    logging_dir='./logs',
    logging_strategy="epoch",
    logging_first_step=True,
    logging_steps=1000,
    save_strategy="epoch",
    seed=args.seed,
    dataloader_drop_last=False,
    dataloader_num_workers=2,
    report_to = 'wandb',
    load_best_model_at_end =True, 
    metric_for_best_model ='accuracy' 
)

if args.losstype == 'dice':
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            criterion = SelfAdjDiceLoss()
            loss = criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    
    trainer = CustomTrainer(
        args=training_args,
        data_collator=batchify,
        model=model,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics
    )
elif args.losstype == 'gaff':
    #! gaussian affinity loss
    from seq_bert import BertForSequenceClassificationWithGAFF

    model = BertForSequenceClassificationWithGAFF.from_pretrained(model_path,
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False,
                                                        problem_type = "imbalanced_single_label_classification",
                                                        classifier_dropout = args.drop_p) #! need check
    model.gaff_sigma = args.gaff_sigma
    model.gaff_margin = args.gaff_margin

    trainer = Trainer(
        args=training_args,
        data_collator=batchify,
        model=model,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics
    )
    

elif args.losstype == 'focal':
    from seq_bert import BertForSequenceClassificationWithFocal
    model = BertForSequenceClassificationWithFocal.from_pretrained(model_path,
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False,
                                                        problem_type = "imbalanced_single_label_classification",
                                                        classifier_dropout = args.drop_p) #! need check

    model.focal_loss_gamma = args.focal_gamma #! hj default: 1

    trainer = Trainer(
        args=training_args,
        data_collator=batchify,
        model=model,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics
    )

elif args.losstype == 'ce':
    trainer = Trainer(
        args=training_args,
        data_collator=batchify,
        model=model,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics
    )

else:
    raise ValueError("not implemented")

set_seed(args.seed)

# evalout = trainer.evaluate(test_ds)
# wandb.log(evalout)

trainer.train()

evalout = trainer.evaluate(test_ds)

wandb.log(evalout)

out = trainer.predict(test_ds)

y_true = out[1]
y_pred = np.argmax(out[0], axis=1)

file_path = os.path.join(output_path, f'clf_out_{args.report_name}.txt')
if os.path.exists(file_path): os.remove(file_path)
out = open(file_path, 'a')

print('-'*10, 'confusion matrix(row:true, column:prediction)', '-'*10, file=out)
print(confusion_matrix(y_true, y_pred), file=out)
print('\n\n', '-'*10, 'classification report', '-'*10, file=out)
print(classification_report(y_true, y_pred, target_names=list(label_dict.keys())), file=out)