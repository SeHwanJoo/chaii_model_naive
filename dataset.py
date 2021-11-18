#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:37:35 2021

@author: sehwan.joo
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
from utils import *


try:
    from torch.cuda import amp
    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

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


def create_folds(data, num_splits):
    data["kfold"] = -1
    kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=9353)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data['language'])):
        data.loc[v_, 'kfold'] = f
    return data


def convert_answers(row):
    return {'answer_start': [row[0]], 'text': [row[1]]}


def prepare_train_features(args, example, tokenizer):
    example["question"] = example["question"].lstrip()
    tokenized_example = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_example.pop("offset_mapping")

    features = []
    for i, offsets in enumerate(offset_mapping):
        feature = {}

        input_ids = tokenized_example["input_ids"][i]
        attention_mask = tokenized_example["attention_mask"][i]

        feature['input_ids'] = input_ids
        feature['attention_mask'] = attention_mask
        feature['offset_mapping'] = offsets

        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_example.sequence_ids(i)

        ## for validation
        feature["example_id"] = example['id'] 
        feature['sequence_ids'] = [0 if i is None else i for i in tokenized_example.sequence_ids(i)] 
        feature['context'] = example["context"]
        feature['question'] = example["question"]
        feature['hindi_tamil'] = 0 if example["language"]=='hindi' else 1 
        sample_index = sample_mapping[i]
        answers = example["answers"]
        if len(answers["answer_start"]) == 0:
            feature["start_position"] = cls_index
            feature["end_position"] = cls_index
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
    
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
    
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                feature["start_position"] = cls_index
                feature["end_position"] = cls_index
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                feature["start_position"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                feature["end_position"] = token_end_index + 1

        features.append(feature)
    return features


class DatasetRetriever(Dataset):
    def __init__(self, features, mode='train'):
        super(DatasetRetriever, self).__init__()
        self.features = features
        self.mode = mode
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, item):   
        feature = self.features[item]
        if self.mode == 'train':
            return {
                'input_ids':torch.tensor(feature['input_ids'], dtype=torch.long),
                'attention_mask':torch.tensor(feature['attention_mask'], dtype=torch.long),
                'offset_mapping':torch.tensor(feature['offset_mapping'], dtype=torch.long),
                'start_position':torch.tensor(feature['start_position'], dtype=torch.long),
                'end_position':torch.tensor(feature['end_position'], dtype=torch.long)
            }
        else:
            if self.mode=='valid':
                return {
                    'input_ids':torch.tensor(feature['input_ids'], dtype=torch.long),
                    'attention_mask':torch.tensor(feature['attention_mask'], dtype=torch.long),
                    'offset_mapping':feature['offset_mapping'],
                    'sequence_ids':feature['sequence_ids'],
                    'start_position':torch.tensor(feature['start_position'], dtype=torch.long),
                    'end_position':torch.tensor(feature['end_position'], dtype=torch.long),
                    'example_id':feature['example_id'],
                    'context': feature['context'],
                    'question': feature['question']
                }
            else:
                return {
                    'input_ids':torch.tensor(feature['input_ids'], dtype=torch.long),
                    'attention_mask':torch.tensor(feature['attention_mask'], dtype=torch.long),
                    'offset_mapping':feature['offset_mapping'],
                    'sequence_ids':feature['sequence_ids'],
                    'id':feature['example_id'],
                    'context': feature['context'],
                    'question': feature['question']
                }      
        
def make_loader(
    args, data, 
    tokenizer, fold
):
    train_set, valid_set = data[data['kfold']!=fold], data[data['kfold']==fold].reset_index(drop=True)
    
    train_features, valid_features = [[] for _ in range(2)]
    for i, row in train_set.iterrows():
        train_features += prepare_train_features(args, row, tokenizer)
    for i, row in valid_set.iterrows():
        valid_features += prepare_train_features(args, row, tokenizer)

    train_dataset = DatasetRetriever(train_features, mode="train")
    valid_dataset = DatasetRetriever(valid_features, mode="valid")
    print(f"Num examples Train= {len(train_dataset)}, Num examples Valid={len(valid_dataset)}")
    
    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler, #wsampler
        num_workers=0,
        pin_memory=True,
        drop_last=False 
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size, 
        sampler=valid_sampler,
        num_workers=0,
        pin_memory=True, 
        drop_last=False
    )

    return train_dataloader, valid_dataloader, valid_features, valid_set
