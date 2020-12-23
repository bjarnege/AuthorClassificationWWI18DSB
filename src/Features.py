# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:43:40 2020

@author: Bjarne Gerdes
"""

import re
import nltk.data
import numpy as np


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
columns = ["Text length", "Number URLs", "Number mails",\
          "Uppercase ratio", "Lowercase ratio", "Number ratio", "Symbol ratio",\
          "Average letters per word", "Variance of letters per word", "Unique words ratio",\
          "Average letters per sentence", "Variance of letters per sentence",\
          "Average words per sentence", "Variance of words per sentence",\
          "Maximal uppercase ratio per sentence", "Length of the maximal uppercase ratio sentence"]


def buildFeatures(text):
    text_split = text.split()
    len_text = len(text)
    sentence_split = tokenizer.tokenize(text)
    
    # find the number of urls in the text
    keywords = ["urlLink","http","www"]
    nb_urls = sum((any(keyword in pattern for keyword in keywords))\
               for pattern in text.split())
    # find the number of mails in the text
    nb_mails = len(re.findall(r"([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+|\bmail\b)"\
                      ,text))
    
    # find the number of dates in the text
#    nb_dates = findDates(text)
     
    # find characteristics about the usage of letters, numbers and symbols
    uppercase_ratio = len(re.findall(r'[A-Z]', text))/len_text
    lowercase_ratio = len(re.findall(r'[a-z]', text))/len_text
    number_ratio = len(re.findall(r'[0-9]', text))/len_text
    symbol_ratio = len(re.findall(r'[$-/:-?{-~!"^_`\[\]]', text))/len_text

    # find characteristics about the letters per word
    sentence_len_word = [len(word) for word in text_split]
    avg_letters_per_word = np.mean(sentence_len_word)
    var_letters_per_word = np.var(sentence_len_word)
    unique_words_ratio = len(set(text_split))/len(text_split)

    # find characteristics about the letters per sentence
    sentence_len_list = [len(sentence) for sentence in sentence_split]
    avg_letters_per_sentence = np.mean(sentence_len_list)
    var_letters_per_sentence = np.var(sentence_len_list)
    
    # find characteristics about the words per sentence
    words_per_sentence_len_list = [len(sentence.split()) for sentence in sentence_split]
    avg_words_per_sentence = np.mean(words_per_sentence_len_list)
    var_words_per_sentence = np.var(words_per_sentence_len_list)
    
    # find the trumps
    uppercase_per_sentence_ratio = [len(re.findall(r'[A-Z]', sentence))/len(sentence)\
                                    for sentence in sentence_split]
    max_sentence_uppercase_ratio = max(uppercase_per_sentence_ratio)
    max_sentence_uppercase_len = len(sentence_split[uppercase_per_sentence_ratio.index(max_sentence_uppercase_ratio)])
    
    features = [len_text, nb_urls, nb_mails,\
               uppercase_ratio, lowercase_ratio, number_ratio, symbol_ratio,\
               avg_letters_per_word, var_letters_per_word, unique_words_ratio,\
               avg_letters_per_sentence, var_letters_per_sentence,\
               avg_words_per_sentence, var_words_per_sentence,\
               max_sentence_uppercase_ratio, max_sentence_uppercase_len]
    
    return dict([(k,v) for v,k in zip(features,columns)])
           