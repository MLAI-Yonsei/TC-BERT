import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
import pandas as pd
import numpy as np

import random
import os
from typing import Union, List, Dict
import argparse
import pdb

# run arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="user-define random seed")
parser.add_argument("--model", type=str, default='best', help="best model or last model")
args = parser.parse_args()

# path
data_path: str = './data/inference_sampleset.csv'
vocab_path: str = './vocab'
result_path: str = './result'

last_model: str = './model/checkpoint-12300/'
best_model: str = './model/checkpoint-984/'

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

# build dataset (dictionary type => id : [sent1, sent2, ..., ])
df = pd.read_csv(data_path, encoding='cp949')

# possible_labels = ['가설 설정', '기술 정의', '기술동향','기술의 파급효과', 
#                    '기술의 필요성', '대상 데이터', '데이터처리', '문제 정의', 
#                    '성능/효과', '시장동향', '이론/모형', '제안 방법', '후속연구']
possible_labels = ['성능/효과', '제안 방법', '대상 데이터', '문제 정의', 
                   '이론/모형', '후속연구', '기술 정의','데이터처리', '가설 설정', 
                   '시장동향', '기술의 파급효과', '기술동향', '기술의 필요성']

label_dict = {}
label2id = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
    label2id[index] = possible_label

# intialize tokenizer and load pre-trained TechBERT
tokenizer = BertTokenizerFast.from_pretrained(vocab_path, do_lower_case=False, model_max_length=128)
model_path = best_model if args.model == 'best' else last_model
model = BertForSequenceClassification.from_pretrained(model_path, 
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# sentence classification with techBERT
inputs = tokenizer(list(df.text), add_special_tokens=True, truncation=True, padding=True, max_length = 128, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax(dim=1).numpy()
    prediction = [label2id[i] for i in predicted_class_id]

# result
compare_df = pd.DataFrame({'text':df.text,'true':df.tag,'pred':prediction})
if os.path.isdir(result_path): pass
else: os.mkdir(result_path)
file_path = os.path.join(result_path, f'classification_result.csv')
if os.path.exists(file_path): os.remove(file_path)
compare_df.to_csv(file_path, index=False, encoding="utf-8-sig")

print(f'acc: {sum(df.tag == prediction) / len(df)}')