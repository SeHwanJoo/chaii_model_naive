#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:40:42 2021

@author: nuvilabs
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
gc.enable()
import math
import json
import time
import random
import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn import model_selection

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
from torch.utils.data.distributed import DistributedSampler
from train import run
from dataset import *

try:
    from torch.cuda import amp
    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False
import argparse

parser = argparse.ArgumentParser(description="let's ride")
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--doc-stride', type=int, default=135)
parser.add_argument('--max-length', type=int, default=400)
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--decay-name', type=str, default='linear-warmup')
parser.add_argument('--seed', type=int, default=9353)
# parser.add_argument('--model-name', type=str, default='/data/Dataset/sehwan/ritm/weight/pretrain/mutil')
# parser.add_argument('--model-name', type=str, default='/data/Dataset/sehwan/ritm/weight/pretrain/roberta-large-wechsel-hindi')
parser.add_argument('--model-name', type=str, default='deepset/xlm-roberta-large-squad2')
arg_parser = parser.parse_args()

import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    logging,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)
logging.set_verbosity_warning()
logging.set_verbosity_error()


class Config:
    # model
    model_type = 'xlm_roberta'
    # [
    # bert-large-uncased-whole-word-masking-finetuned-squad, 
    # google/rembert
    # deepset/roberta-large-squad2, 
    # deepset/xlm-roberta-large-squad2,
    # deepset/bert-large-uncased-whole-word-masking-squad2
    # google/muril-large-cased
    # ]
    model_name_or_path = "deepset/bert-large-uncased-whole-word-masking-squad2"
    config_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
    fp16 = False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2
    fold = 0

    # tokenizer
    tokenizer_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
    max_seq_length = 384
    doc_stride = 128

    # train
    epochs = 10
    train_batch_size = 4
    eval_batch_size = 256

    # optimizer
    optimizer_type = 'AdamW'
    learning_rate = 1.5e-5
    weight_decay = 1e-2
    epsilon = 1e-8
    max_grad_norm = 5.0

    # scheduler
    decay_name = 'linear-warmup'
    warmup_ratio = 0.1

    # logging
    logging_steps = 100
    
    #early stopping patience
    patience = 2

    # evaluate
    output_dir = f'/data/Dataset/sehwan/ritm/weight/{model_name_or_path}'
    seed = 9353
    
    # loss
    loss = 'dice'
    trans = False
    multi_start = False
    
    def update(self, args):
        self.model_name_or_path = args.model_name
        self.config_name = args.model_name
        self.tokenizer_name = args.model_name
        self.max_seq_length = args.max_length
        self.doc_stride = args.doc_stride
        self.train_batch_size = args.batch_size
        self.eval_batch_size = args.batch_size
        self.fold = args.test
        self.epochs = args.epoch
        self.decay_name = args.decay_name
        self.seed = args.seed
        if '/data/Dataset' in args.model_name:
            self.output_dir = f'{args.model_name}/{args.seed}/nofold'
        else:
            self.output_dir = f'/data/Dataset/sehwan/ritm/weight/{args.model_name}/{args.seed}/nofold'
    
# train = pd.read_csv('dataset/train.csv')
train = pd.read_csv('dataset/chaii0917.csv')
test = pd.read_csv('dataset/test.csv')
external_mlqa = pd.read_csv('dataset/mlqa_hindi.csv')
external_xquad = pd.read_csv('dataset/xquad.csv')
external_trans = pd.read_csv('dataset/squad_translated_tamil.csv')
squad_hi = pd.read_csv('dataset/squad_hi.csv')
squad_ta = pd.read_csv('dataset/squad_ta.csv')
squad_tamilQA = pd.read_csv('dataset/squad_tamilQA.csv')
XQA_tamil_dev_query = pd.read_csv('dataset/XQA_tamil_dev_query.csv')
XQA_tamil_dev = pd.read_csv('dataset/XQA_tamil_dev.csv')
XQA_tamil_test_query = pd.read_csv('dataset/XQA_tamil_test_query.csv')
XQA_tamil_test = pd.read_csv('dataset/XQA_tamil_test.csv')
ext0917 = pd.read_csv('dataset/ext0917.csv')
# train = pd.read_csv('dataset/train_en.csv')
external_train = pd.concat([
    # external_mlqa, 
    # external_xquad, 
    ext0917,
    # squad_hi,
    # squad_ta,
    # squad_tamilQA,
    # XQA_tamil_dev_query,
    # XQA_tamil_dev,
    # XQA_tamil_test_query,
    # XQA_tamil_test
    # external_trans
    ])



train = create_folds(train, num_splits=5)
external_train["kfold"] = -1
external_train['id'] = list(np.arange(1, len(external_train)+1))
train = pd.concat([train, external_train]).reset_index(drop=True)

hindi_list = []
tamil_list = []
for index, row in train[['language']].iterrows():
    if row[0]=='hindi':
        hindi_list.append(True)
    else:
        tamil_list.append(True)
print(len(hindi_list))
print(len(tamil_list))

languages = ['en', 'fr', 'th', 'tr', 'ur', 'ru', 'bg', 'de', 'ar', 'zh-cn', 'sw', 'vi', 'es', 'el']

def back_translate(row, PROB = 1):

    sequence_list = []
    sequence_list.append(row[0])
    sequence_list.append(row[1])
    sequence_list.append(row[2])
    org_lang = row[3]
    #instantiate translator
    translator = Translator()
    
    #store original language so we can convert back
    
    #randomly choose language to translate sequence to  
    random_lang = np.random.choice([lang for lang in languages if lang is not org_lang])
    output_sequence_list = []
    for sequence in sequence_list:
        if org_lang in languages:
            #translate to new language and back to original
            translated = translator.translate(sequence, dest = random_lang).text
            #translate back to original language
            translated_back = translator.translate(translated, dest = org_lang).text
        
            #apply with certain probability
            if np.random.uniform(0, 1) <= PROB:
                output_sequence = translated_back
            else:
                output_sequence = sequence
                
        #if detected language not in our list of languages, do nothing
        else:
            output_sequence = sequence
        output_sequence_list.append(output_sequence)
    return output_sequence_list

if Config().trans:
    context_list = []
    question_list = []
    answer_text_list = []
    trans_lan_list = []
    answer_start_list = []
    kfold_list = []
    id_list = []
    lan_list = []
    
    for index, row in train[['context', 'question', 'answer_text', 'language','kfold', 'id']].iterrows():
        context_, question_, answer_text_ = back_translate(row)
        if answer_text_ not in context_:
            print(answer_text_)
        else:
            context_list.append(context_)
            question_list.append(question_)
            answer_text_list.append(answer_text_)
            answer_start_list.append(context_.find(answer_text_))
            kfold_list.append(row[4])
            id_list.append(row[5])
            lan_list.append(row[3])
    train = pd.DataFrame()
    train['answer_text'] = answer_text_list
    train['context'] = context_list
    train['question'] = question_list
    train['answer_start'] = answer_start_list
    train['kfold'] = kfold_list
    train['id'] = id_list
    train['language'] = lan_list
train['answers'] = train[['answer_start', 'answer_text']].apply(convert_answers, axis=1)

args = Config()
args.update(arg_parser)
print();print()
print('-'*50)
print(f'FOLD: {args.fold}')
print('-'*50)
print(args)
run(train, args.fold, args)
