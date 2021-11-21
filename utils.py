#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:36:39 2021

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
from model import *
from dataset import *
import collections
import torch.optim as optim

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

def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus*4) if num_gpus else num_cpus - 1
    return optimal_value

    
def make_model(args):
    config = AutoConfig.from_pretrained(args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = Model(args.model_name_or_path, config=config)
    return config, tokenizer, model

def make_optimizer(args, model):
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.optimizer_type == "AdamW":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.epsilon,
            correct_bias=True
        )
        return optimizer

def make_scheduler(
    args, optimizer, 
    num_warmup_steps, 
    num_training_steps
):
    if args.decay_name == "cosine-warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif args.decay_name == "linear-warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.1)
    return scheduler    


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    
    return float(len(c)) / (len(a) + len(b) - len(c))


def postprocess_qa_predictions(examples, features1, raw_predictions, tokenizer, n_best_size = 20, max_answer_length = 30):
    features = features1
    all_start_logits, all_end_logits = raw_predictions
    
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]
        #print(example['id'],example_index,feature_indices)
        min_null_score = None
        valid_answers = []
        
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            offset_mapping = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]    
        
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = best_answer["text"]
        
        
    return predictions