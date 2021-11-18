#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 19:13:56 2021

@author: sehwan.joo
"""
from train import init_training
import pandas as pd
from utils import create_folds, postprocess_qa_predictions, jaccard
from dataset import convert_answers
from model import Evaluator
import numpy as np
import torch

class Config:
    # model
    model_type = 'xlm_roberta'
    model_path = '/data/Dataset/sehwan/ritm/weight/pretrain/mutil/finetuning_9353/all/epoch-1-fold-0'
    # [
    # bert-large-uncased-whole-word-masking-finetuned-squad, 
    # google/rembert
    # deepset/roberta-large-squad2, 
    # deepset/xlm-roberta-large-squad2,
    # deepset/bert-large-uncased-whole-word-masking-squad2
    # google/muril-large-cased
    # ]
    model_name_or_path = "/data/Dataset/sehwan/ritm/weight/pretrain/mutil"
    config_name = "/data/Dataset/sehwan/ritm/weight/pretrain/mutil"
    fp16 = False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2
    fold = 0

    # tokenizer
    tokenizer_name = "/data/Dataset/sehwan/ritm/weight/pretrain/mutil"
    max_seq_length = 400
    doc_stride = 135

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
    
    
train = pd.read_csv('dataset/train.csv')
train = create_folds(train, num_splits=5)
train['answers'] = train[['answer_start', 'answer_text']].apply(convert_answers, axis=1)

args = Config()
fold = int(args.model_path[-1])
model, model_config, tokenizer, optimizer, scheduler, train_dataloader, \
        valid_dataloader, result_dict, valid_features, valid_set = init_training(args, train, fold)
model.load_state_dict(torch.load(args.model_path + '/pytorch_model.bin'))
model.eval()
evaluator = Evaluator(model)
epoch = 0
result_dict, all_outputs_start, all_outputs_end = evaluator.evaluate(
    valid_dataloader, epoch, result_dict
)

def print_jaccard(features, n_best_size, max_answer_length):
    valid_features1 = features.copy()
    valid_preds = postprocess_qa_predictions(valid_set, valid_features1, (all_outputs_start, all_outputs_end), tokenizer, 
                                             n_best_size=n_best_size, max_answer_length=max_answer_length)
    valid_set['PredictionString'] = valid_set['id'].map(valid_preds)
    valid_set['jaccard'] = valid_set[['answer_text','PredictionString']].apply(lambda x: jaccard(x[0],x[1]), axis=1)
    tami_valid_set = valid_set[valid_set['language']=='tamil']
    hindi_valid_set = valid_set[valid_set['language']=='hindi']
    print(n_best_size, max_answer_length)
    print(f'all: {np.mean(valid_set.jaccard)}')
    print(f'tamil: {np.mean(tami_valid_set.jaccard)}')
    print(f'hindi: {np.mean(hindi_valid_set.jaccard)}')
    
print_jaccard(valid_features, 20, 35)
print_jaccard(valid_features, 20, 40)
print_jaccard(valid_features, 20, 45)
print_jaccard(valid_features, 20, 50)
print_jaccard(valid_features, 5, 30)
print_jaccard(valid_features, 10, 30)
print_jaccard(valid_features, 15, 30)
print_jaccard(valid_features, 25, 30)
print_jaccard(valid_features, 30, 50)
print_jaccard(valid_features, 40, 50)
print_jaccard(valid_features, 50, 50)
print_jaccard(valid_features, 20, 50)




# print(hindi_valid_set.jaccard)

result_dict['jaccard'].append(np.mean(valid_set.jaccard))