#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:38:14 2021

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
# from utils import *
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


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val
            
            
class Model(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(Model, self).__init__()
        self.config = config
        self.xlm_roberta = AutoModel.from_pretrained(modelname_or_path, config=config)
        # self.layer1 = nn.Conv1d(400, 400, 3, padding=1)
        # self.layer2 = nn.Conv1d(400, 400, 3, padding=1)
        # self.layer3 = nn.Conv1d(400, 400, 3, padding=1)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)
        self._init_weights(self.qa_outputs)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        # token_type_ids=None
    ):
        outputs = self.xlm_roberta(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]
        # sequence_layer1 = self.layer1(sequence_output)
        # sequence_layer1 = nn.GELU()(sequence_layer1)
        # sequence_layer2 = self.layer1(sequence_layer1)
        # sequence_layer2 = nn.GELU()(sequence_layer2)
        # sequence_layer3 = self.layer1(sequence_layer2)
        # sequence_layer3 = nn.GELU()(sequence_layer3)
        
        # sequence_output = sequence_layer1 + sequence_layer2 + sequence_layer3
        
        # sequence_output = self.dropout(sequence_output)
        qa_logits = self.qa_outputs(sequence_output)
        
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
    
        return start_logits, end_logits
    

def loss_fn(preds, labels):
    start_preds, end_preds = preds
    start_labels, end_labels = labels
    start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_preds, start_labels)
    end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels) 
    total_loss = start_loss * 0.5 + end_loss * 0.5
    return total_loss

def pos_weight(pred_tensor, pos_tensor, neg_weight=1, pos_weight=1):
    # neg_weight for when pred position < target position
    # pos_weight for when pred position > target position
    gap = torch.argmax(pred_tensor, dim=1) - pos_tensor
    gap = gap.type(torch.float32)
    return torch.where(gap < 0, -neg_weight * gap, pos_weight * gap).detach()


def loss_fn_modified(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss(reduce='none') # do reduction later
    
    start_loss = loss_fct(start_logits, start_positions) * pos_weight(start_logits, start_positions, 1, 1)
    end_loss = loss_fct(end_logits, end_positions) * pos_weight(end_logits, end_positions, 1, 1)
    
    start_loss = torch.mean(start_loss)
    end_loss = torch.mean(end_loss)
    
    total_loss = (start_loss + end_loss)
    return total_loss



class Trainer:
    def __init__(
        self, model, tokenizer, 
        optimizer, scheduler
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(
        self, args, 
        train_dataloader, 
        epoch, result_dict
    ):
        count = 0
        losses = AverageMeter()
        
        self.model.zero_grad()
        self.model.train()
        
        # fix_all_seeds(args.seed)
        
        for batch_idx, batch_data in enumerate(train_dataloader):
            input_ids, attention_mask, targets_start, targets_end = \
                batch_data['input_ids'], batch_data['attention_mask'], \
                    batch_data['start_position'], batch_data['end_position']
            
            input_ids, attention_mask, targets_start, targets_end = \
                input_ids.cuda(), attention_mask.cuda(), targets_start.cuda(), targets_end.cuda()

            outputs_start, outputs_end = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            loss = loss_fn((outputs_start, outputs_end), (targets_start, targets_end))
            # loss = loss_fn_modified(outputs_start, outputs_end, targets_start, targets_end)
            loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            count += input_ids.size(0)
            losses.update(loss.item(), input_ids.size(0))

            # if args.fp16:
            #     torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
            # else:
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

#             if batch_idx % args.gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
#                 self.optimizer.step()
#                 self.scheduler.step()
#                 self.optimizer.zero_grad()


            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            
            if (batch_idx % args.logging_steps == 0) or (batch_idx+1)==len(train_dataloader):
                _s = str(len(str(len(train_dataloader.sampler))))
                ret = [
                    ('Epoch: {:0>2} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(train_dataloader.sampler), 100 * count / len(train_dataloader.sampler)),
                    'Train Loss: {: >4.5f}'.format(losses.avg),
                ]
                print(', '.join(ret))

        result_dict['train_loss'].append(losses.avg)
        return result_dict
    
    
class Evaluator:
    def __init__(self, model):
        self.model = model
    
    def save(self, result, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/result_dict.json', 'w') as f:
            f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))

    def evaluate(self, valid_dataloader, epoch, result_dict):
        losses = AverageMeter()
        all_outputs_start, all_outputs_end = [], []
        for batch_idx, batch_data in enumerate(valid_dataloader):
            self.model = self.model.eval()
            input_ids, attention_mask, targets_start, targets_end = \
                batch_data['input_ids'], batch_data['attention_mask'], \
                    batch_data['start_position'], batch_data['end_position']
            
            input_ids, attention_mask, targets_start, targets_end = \
                input_ids.cuda(), attention_mask.cuda(), targets_start.cuda(), targets_end.cuda()
            
            with torch.no_grad():            
                outputs_start, outputs_end = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                all_outputs_start.append(outputs_start.cpu().numpy().tolist())
                all_outputs_end.append(outputs_end.cpu().numpy().tolist())
                
                loss = loss_fn((outputs_start, outputs_end), (targets_start, targets_end))
                losses.update(loss.item(), input_ids.size(0))
                
        all_outputs_start = np.vstack(all_outputs_start)
        all_outputs_end = np.vstack(all_outputs_end)
                
        print('----Validation Results Summary----')
        print('Epoch: [{}] Valid Loss: {: >4.5f}'.format(epoch, losses.avg))
        result_dict['val_loss'].append(losses.avg)     
        return result_dict, all_outputs_start, all_outputs_end
    
    
