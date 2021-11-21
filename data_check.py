
"""
Created on Fri Nov  5 14:48:05 2021

@author: sehwan.joo
"""
import pandas as pd
import re
train = pd.read_csv('dataset/train.csv')

answer_start_list = []
for index, row in train.iterrows():
    # print(row['context'][row['answer_start']])
    # print(row['answer_text'])
    index_list = []
    context = row['context']
    answer_text = row['answer_text']
    prev_index = 0
    while answer_text in context[prev_index:] and prev_index < len(context)-1:
        index = context[prev_index:].find(answer_text) + prev_index
        index_list.append(index)
        prev_index = index + len(answer_text)
    answer_start_list.append(index_list)

train['answer_start_list'] = answer_start_list
train.to_csv('dataset/train_list.csv', index=False)