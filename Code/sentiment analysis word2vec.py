# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:51:02 2021

@author: Shaked Abeahamy

sentiment analysis with word embedding, based on pre-trained embedding
Data: Exceed.AI emails


the NLP model train on 61K mails of exceed and the vocabulary is more than 11K words
train data is 1K tagged emails
"""

# =============================================================================
# packages
# =============================================================================

## for data
import json
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for processing
import re

## for bag-of-words
## for explainer
#from lime import lime_text
## for word embedding
import gensim
import gensim.downloader as gensim_api
import pickle
from gensim.models import Word2Vec
import random

# =============================================================================
# Data
# =============================================================================
TaggedData = pd.read_csv(r"C:\Users\Owner\Desktop\פרויקט גמר\data\taggedData.csv")
TaggedData.columns = ["id", "email","class"]
classes = TaggedData['class'].values.tolist()
new_class = []
for ec in classes:
    pattern = re.compile('; ')
    ec = pattern.sub('_', ec) ##Identifies a word that begins with an Internet prefix and replaces with a placed sign
    new_class.append(ec)
TaggedData['class'] = new_class

# Data after processing
with open('listfile.data', 'rb') as filehandle:
    # read the data as binary data stream
    emails_after_process_list = pickle.load(filehandle)
# initiate model
W2V_model = Word2Vec(emails_after_process_list, min_count=4)

#import tagged data after preprocessing
with open('tagged_after_process.data', 'rb') as filehandle:
    # read the data as binary data stream
    tagged_after_process_list = pickle.load(filehandle)

# deal with varying length
length = []
for x in tagged_after_process_list:
    length.append(len(x))
max(length)

#remove outliers - mails with more than 300 words
for i in range(len(length)-1):
    if length[i] > 300 :
       print(f" remove email number {i} with length {length[i]}")
       del length[i]
       del tagged_after_process_list[i]

# split the mails to val and train
X_train_emails,X_val_emails = [],[] 
for i in tagged_after_process_list:
    x = random.uniform(0, 1)
    if x < 0.8:
        X_train_emails.append(i)
    else:
        X_val_emails.append(i)


embeddings_index = {}
for w in W2V_model.wv.vocab.keys():
    embeddings_index[w] = W2V_model.wv[w]
    
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=1100)
tokenizer.fit_on_texts(X_train_emails) # every word represrnt by a num
sequences = tokenizer.texts_to_sequences(X_train_emails)


# "padding" evry sentence less than 300 dim
x_train_seq = pad_sequences(sequences, maxlen=300)
sequences_val = tokenizer.texts_to_sequences(X_val_emails)
x_val_seq = pad_sequences(sequences_val, maxlen=300)

num_words = 10000
embedding_matrix = np.zeros((num_words, 100))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
# =============================================================================
# Convolutional Neural Network
# =============================================================================
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
structure_test = Sequential()
e = Embedding(num_words, 100, input_length=300)
structure_test.add(e)
structure_test.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
structure_test.summary()        
