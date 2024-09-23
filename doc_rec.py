import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
import pandas as pd
import numpy as np

import random
import os
import pickle
from typing import Union, List, Dict
from collections import Counter, defaultdict
import argparse
import pdb 

# run arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="user-define random seed")
parser.add_argument("--src_id", type=int, default=0, help="source document id")
parser.add_argument("--topk", type=int, default=3, help="# of recommended target docs")
parser.add_argument("--run_scratch", type=int, default=0, help="1 or 0 run from scratch or not")
parser.add_argument("--model", type=str, default='best', help="best model or last model")
args = parser.parse_args()

# path 
data_path: str = './data/etri4rec_v2.csv'
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
# and nested dictionary for analysis on recommendation result
# {document id : {sentence id : text}} 
print("build data dicitionary ...")
df = pd.read_csv(data_path, encoding='cp949')
original_doc_idx = df.doc_id.unique()
tran2ori = {k:v for k, v in enumerate(original_doc_idx)}
print(f"# of documents: {len(original_doc_idx)}")

data_dict = {}
nested_data_dict_t = defaultdict(lambda: defaultdict(int))
for i, doc in enumerate(original_doc_idx):
    data_dict[i] = list(df[df.doc_id == doc].text)
    for j, sent in enumerate(np.where(df.doc_id == doc)[0]):
        nested_data_dict_t[i][j] = df.iloc[sent].text
nested_data_dict = {k: {k2: v2 for k2, v2 in v.items()} for k, v in nested_data_dict_t.items()}

# label list and label-to-id dictionary
possible_labels = ['가설 설정', '기술 정의', '기술동향','기술의 파급효과', 
                   '기술의 필요성', '대상 데이터', '데이터처리', '문제 정의', 
                   '성능/효과', '시장동향', '이론/모형', '제안 방법', '후속연구']
# possible_labels = ['성능/효과', '제안 방법', '대상 데이터', '문제 정의', '이론/모형', '후속연구', '기술 정의',
#                    '데이터처리', '가설 설정', '시장동향', '기술의 파급효과', '기술동향', '기술의 필요성']

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

# Recommendation System
class Recommender(object):
    def __init__(self, model, tokenizer, dataset) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.ds = dataset
        self.n_docs = len(dataset)
        self.pool_emb = {}
        self.pool_logits = {}
        self.pool_pred = {}
    
    def encode_src(self, src_id):
        '''
            source 문서 하나를 받아 인코딩하고 관련정보 저장
        '''
        # raise error when the inputed id is out of the docs range
        if src_id >= self.n_docs:
            raise ValueError(f'src_id must be less than {self.n_docs}')

        inputs = tokenizer(self.ds[src_id], add_special_tokens=True, padding='max_length', return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
            logits, emb = out[0], out[1][12][:,0,:].squeeze()
            # (# of sentence, # of categories), (# of sentence, emb dim)
            # (N, C), (N, D)
            predicted_class_id = logits.argmax(dim=1).numpy()
            prediction = [label2id[i] for i in predicted_class_id]

        # source document info.
        self.src_id = src_id
        self.src_logits = logits
        self.src_emb = emb
        self.src_pred = prediction
        
    def encode_pool(self):
        '''
            전체 데이터셋을 인코딩하고 딕셔너리 형태로 정보저장 
            {id: emb}, {id: logit}, {id: pred}
        '''
        for id in range(self.n_docs):
            inputs = tokenizer(self.ds[id], add_special_tokens=True, truncation=True, padding='max_length', return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, return_dict=True)
                logits, emb = out[0], out[1][12][:,0,:].squeeze()
                # (# of sentence, # of categories), (# of sentence, emb dim)
                # (N, C), (N, D)
                predicted_class_id = logits.argmax(dim=1).numpy()
                prediction = [label2id[i] for i in predicted_class_id]
            
            # build document pool
            self.pool_emb[id] = emb
            self.pool_logits[id] = logits
            self.pool_pred[id] = prediction
    
    def src_checker(self):
        '''
            source 문서에 포함된 문장들에 대해 모델이 예측한 태그 반환
        '''
        src_tag_dic = Counter(self.src_pred)
        return list(src_tag_dic.keys())

    def recommendation(self, tag, src_id=None, topk=1):
        '''
            주어진 src 문서의 tag 문장들에 대한 TechBERT의 평균 임베딩을 기준으로,
            pool 문서들 중 tag 문장들의 TechBERT 평균 임베딩이 가장 유사한 문서들의 id를 반환
            + 거기에 더하여, 왜 그런 추천을 했는지 해석하기 위해 src 문서와 추천된 문서들에서 
              해당 tag로 예측된 문장들 인덱스도 함께 반환
        '''
        if src_id is not None:
            self.encode_src(src_id)
        
        # 1. get query sentence indicies and make query vector
        src_tag_idx = [i for i, pred in enumerate(self.src_pred) if pred == tag]
        query_embs = self.src_emb[src_tag_idx] 
        query_emb = query_embs.mean(dim=0).unsqueeze(0) #! (1, D)

        # 2. build embedding matrix w.r.t. query tag
        key_embs = torch.zeros(self.n_docs, query_emb.shape[1]) #! (N, D)
        for id in range(self.n_docs):
            tag_idx_i = [i for i, pred in enumerate(self.pool_pred[id]) if pred == tag]
            if len(tag_idx_i) != 0:
                key_embs_i = self.pool_emb[id][tag_idx_i]
                key_embs[id] = key_embs_i.mean(dim=0).unsqueeze(0) 
            else:
                key_embs[id] = 0 #! doc has no prediction as query tag
        
        # 3. embedding-similarity-based topk target documents
        sim_matrix = self.similarity(query_emb, key_embs) #! (1, N)    
        sim_matrix[0, self.src_id] = -np.inf #! filter the source document
        topk_idx = torch.topk(sim_matrix, k=topk)[1].squeeze()

        # 4. aggregate target indices
        tar_tag_idx = [ [i for i, pred in enumerate(self.pool_pred[id]) if pred == tag] for id in np.array(topk_idx)]
        return topk_idx, src_tag_idx, tar_tag_idx
        
    # define similarity measure
    def similarity(self, query, key, strategy='cos'):
        if strategy == 'cos':
            query      = torch.nn.functional.normalize(query, dim=1)
            key        = torch.nn.functional.normalize(key, dim=1)
            sim_matrix = query @ key.T
        else:
            raise ValueError('not implemented')
        return sim_matrix

# --------------------------------------------------------- doc recommendation procedure
#pdb.set_trace()
rec_sys = Recommender(model, tokenizer, data_dict)
if not args.run_scratch:
    #! load extracted pool embedding 
    #! 사전에 저장해놓은 recommender 클래스 및 문서 임베딩 풀 로드
    print('load recommender ...')
    rec_sys = pickle.load(open('./techbert_rec.pkl', 'rb'))
else:
    #! initially extract pool embedding
    #! TechBERT를 통해 전체 문서들을 encode
    print('encode entire docs ...')
    rec_sys.encode_pool()

    print('save recommender ...')
    pickle.dump(rec_sys, open('./techbert_rec.pkl', 'wb'))

#! TechBERT을 통한 소스문서 인코딩
print('encode source doc ...')
rec_sys.encode_src(src_id = args.src_id)

#! 추천 소스 문서의 각 문장들에 대해 모델이 예측한 태그들을 반환
print('check tags in source doc ...')
src_tags = rec_sys.src_checker()
print(src_tags)

def rec_result(src_id, topk, result_path):
    #! 결과 저장 파일 경로
    if os.path.isdir(result_path): pass
    else: os.mkdir(result_path)
    file_path = os.path.join(result_path, f'rec_for_{src_id}.txt')
    if os.path.exists(file_path): os.remove(file_path)
    out = open(file_path, 'a')
    
    #! TechBERT을 통한 소스문서 인코딩
    print(f'encode source document (id: {src_id})', file=out)
    rec_sys.encode_src(src_id = src_id)

    #! 추천 소스 문서의 각 문장들에 대해 모델이 예측한 태그들을 반환
    print(f'check tags in source doc {src_id} (tran) {tran2ori[src_id]} (ori) ', file=out)
    src_tags = rec_sys.src_checker()
    # src_tags = ['기술동향', '필요성']
    print(src_tags, file=out) 

    for search_tag in src_tags:
        #! 소스 문서에 대해 확보된 각 태그들에 대하여 전체 대상문서 풀로부터,
        #! 각 태그 기준으로 가장 유사한 topK 타겟 문서들의 인덱스와, 
        #! 결과 해석을 위해 소스/타겟 문서들 내에서 해당 태그로 분류된 모든 문장들의 인덱스 및 실제 문장을 반환
        print('\n=====================================================', file=out)
        print(f'recommend target docs that has similar "{search_tag}" semantic with source doc', file=out)
        rec_docs, src_idx, tar_idx = rec_sys.recommendation(tag=search_tag, topk=topk)
        print(f"recommended TOP-{topk} docs: {rec_docs} (tran), {[tran2ori[i] for i in rec_docs.tolist()]} (ori)", file=out)
        print(f"(source) predicted as {search_tag}: {src_idx}", file=out)
        print("(source) query sentences:", file=out)
        for idx in src_idx:
            print(nested_data_dict[src_id][idx], file=out)

        print(f"\n(target) predicted as {search_tag}: {tar_idx}", file=out)
        for i, tar_doc in enumerate(rec_docs.tolist()):
            print(f"(target doc {tar_doc}) key sentences:", file=out)
            for sent_id in tar_idx[i]:
                print(nested_data_dict[tar_doc][sent_id], file=out)


rec_result(args.src_id, args.topk, result_path=result_path)