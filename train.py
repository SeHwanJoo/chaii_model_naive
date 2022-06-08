#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:37:29 2021

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
# from model import *
from utils import *
from dataset import *

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
    
def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def init_training(args, data, fold):
    fix_all_seeds(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # model
    model_config, tokenizer, model = make_model(args)
    if torch.cuda.device_count() >= 1:
        print('Model pushed to {} GPU(s), type {}.'.format(
            torch.cuda.device_count(), 
            torch.cuda.get_device_name(0))
        )
        model = model.cuda() 
    else:
        raise ValueError('CPU training is not supported')
    
    # data loaders
    train_dataloader, valid_dataloader, valid_features, valid_set = make_loader(args, data, tokenizer, fold)

    # optimizer
    optimizer = make_optimizer(args, model)

    # scheduler
    num_training_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.epochs
    if args.warmup_ratio > 0:
        num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    else:
        num_warmup_steps = 0
    print(f"Total Training Steps: {num_training_steps}, Total Warmup Steps: {num_warmup_steps}")
    scheduler = make_scheduler(args, optimizer, num_warmup_steps, num_training_steps)

    # mixed precision training with NVIDIA Apex
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    
    result_dict = {
        'epoch':[], 
        'train_loss': [], 
        'val_loss' : [], 
        'best_val_loss': np.inf,
        'best_jaccard': 0.0,
        'jaccard': [],
        'tamil_jaccard': [],
        'hindi_jaccard': [],
        'best_loss': np.inf
    }

    return (
        model, model_config, tokenizer, optimizer, scheduler, 
        train_dataloader, valid_dataloader, result_dict, valid_features, valid_set
    )

def run(data, fold, args):
    model, model_config, tokenizer, optimizer, scheduler, train_dataloader, \
        valid_dataloader, result_dict, valid_features, valid_set = init_training(args, data, fold)
    
    trainer = Trainer(model, tokenizer, optimizer, scheduler)
    evaluator = Evaluator(model)
    
    patience =args.patience
    
    train_time_list = []
    valid_time_list = []
    
    

    for epoch in range(args.epochs):
        result_dict['epoch'].append(epoch)
        
        # Train
        torch.cuda.synchronize()
        tic1 = time.time()
        result_dict = trainer.train(
            args, train_dataloader, 
            epoch, result_dict
        )
        torch.cuda.synchronize()
        tic2 = time.time() 
        train_time_list.append(tic2 - tic1)
        
        output_dir = os.path.join(args.output_dir, f"epoch-{epoch}-fold-{fold}")
        
    #     # Evaluate
    #     torch.cuda.synchronize()
    #     tic3 = time.time()
    #     result_dict, all_outputs_start, all_outputs_end = evaluator.evaluate(
    #         valid_dataloader, epoch, result_dict
    #     )
    #     torch.cuda.synchronize()
    #     tic4 = time.time() 
    #     valid_time_list.append(tic4 - tic3)
            
        
    #     valid_features1 = valid_features.copy()
    #     valid_preds = postprocess_qa_predictions(valid_set, valid_features1, (all_outputs_start, all_outputs_end), tokenizer)
    #     valid_set['PredictionString'] = valid_set['id'].map(valid_preds)
    #     valid_set['jaccard'] = valid_set[['answer_text','PredictionString']].apply(lambda x: jaccard(x[0],x[1]), axis=1)
    #     result_dict['jaccard'].append(np.mean(valid_set.jaccard))
    #     result_dict['tamil_jaccard'].append(np.mean(valid_set[valid_set['language']=='tamil'].jaccard))
    #     result_dict['hindi_jaccard'].append(np.mean(valid_set[valid_set['language']=='hindi'].jaccard))
    #     print("valid jaccard: ", np.mean(valid_set.jaccard))
    #     # all_jacard_scores.append(np.mean(valid_set.jaccard))
        
        
    #     if result_dict['jaccard'][-1] > result_dict['best_jaccard']:
    #         es_counter=0
    #         print("{} Epoch, Best epoch was updated! Jaccard: {: >4.5f} | Old Jaccard: {: >4.5f}".format(epoch, result_dict['jaccard'][-1], result_dict["best_jaccard"]))
    #         result_dict["best_jaccard"] = result_dict['jaccard'][-1]
    #         result_dict["best_loss"] = result_dict['val_loss'][-1]
    #         os.makedirs(output_dir, exist_ok=True)
    #         torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
    #         model_config.save_pretrained(output_dir)
    #         tokenizer.save_pretrained(output_dir)
    #         print(f"Saving model checkpoint to {output_dir}.")
            
    #     else:
    #         es_counter+=1
    #         print('Valid Loss did not Improve  :(')
    #         print(f'Early stop counter {es_counter} of {patience}')
            
    #     if es_counter == patience:
    #         print()
    #         print('-'*50)
    #         print('!!!!Early stopping!!!!')
    #         print('-'*50)
    #         break
            
    #     print()
        
    # evaluator.save(result_dict, output_dir)
    # evaluator.save(result_dict, output_dir.replace(f'epoch-{epoch}', f'epoch-{epoch-patience}'))
    # print(f"Total Training Time: {np.sum(train_time_list)}secs, Average Training Time per Epoch: {np.mean(train_time_list)}secs.")
    # print(f"Total Validation Time: {np.sum(valid_time_list)}secs, Average Validation Time per Epoch: {np.mean(valid_time_list)}secs.")
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/result_dict.json', 'w') as f:
        f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))
    torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
    model_config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saving model checkpoint to {output_dir}.")

    
