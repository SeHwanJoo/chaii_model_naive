#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:45:18 2021

@author: sehwan
"""

from nltk.corpus import wordnet 
import advertools as adv
import random
import pandas as pd
hindi_stopwords = adv.stopwords['hindi']
tamil_stopwords = adv.stopwords['tamil']
stop_words = list(hindi_stopwords) + list(tamil_stopwords)

def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()
    
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def synonym_replacement(words, n):
    
    words = words.split()
    
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            print(synonym)
            print(len(synonym.split()))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        else:
            print(synonyms)
        
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence

train = pd.read_csv('dataset/train_bg.csv')
word = train['context'][0]
synonym_replacement(word, len(word)//5)
