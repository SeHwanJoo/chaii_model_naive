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
from googletrans import Translator

languages = ['en', 'fr', 'th', 'tr', 'ur', 'ru', 'bg', 'de', 'ar', 'zh-cn', 'sw', 'vi', 'es', 'el']
train = pd.read_csv('dataset/train.csv')

def back_translate(row, PROB = 1, random_lang = 'en'):

    sequence_list = []
    sequence_list.append(row[0])
    sequence_list.append(row[1])
    sequence_list.append(row[2])
    org_lang = row[3]
    #instantiate translator
    translator = Translator()
    
    #store original language so we can convert back
    
    #randomly choose language to translate sequence to  
    # random_lang = np.random.choice([lang for lang in languages if lang is not org_lang])
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

for random_lang in languages:
    context_list = []
    question_list = []
    answer_text_list = []
    trans_lan_list = []
    answer_start_list = []
    id_list = []
    lan_list = []
    
    for index, row in train[['context', 'question', 'answer_text', 'language', 'id']].iterrows():
        context_, question_, answer_text_ = back_translate(row, random_lang=random_lang)
        if answer_text_ not in context_:
            print(answer_text_)
        else:
            context_list.append(context_)
            question_list.append(question_)
            answer_text_list.append(answer_text_)
            answer_start_list.append(context_.find(answer_text_))
            id_list.append(row[4])
            lan_list.append(row[3])
    train = pd.DataFrame()
    train['answer_text'] = answer_text_list
    train['context'] = context_list
    train['question'] = question_list
    train['answer_start'] = answer_start_list
    train['id'] = id_list
    train['language'] = lan_list
    
    train.to_csv(f'./dataset/train_{random_lang}.csv', index=False)
train_df_list = []
for random_lang in languages:
    train_df_list.append(pd.read_csv(f'./dataset/train_{random_lang}.csv'))
train = pd.read_csv('./dataset/train.csv')

train_df_list[0]['answer']
train_df_list[i]['answer_text'][index]

for i, random_lang in languages:
    for index, row in train.iterrows():
        if i==0:
            word = train_df_list[i]['answer_text'][index]
        else:
            if word != train_df_list[i]['answer_text'][index]:
                print('no')
    
