!pip install PyKomoran
import json 
import torch
import torch.nn as nn
from PyKomoran import *

komoran = Komoran("EXP")

with open("SpeechAct_tr.json") as json_file:
    # json_data = {'1': [['user', '아름아 잘 잤니?', 'opening'], ['system', '네, 잘 잤습니다.', 'opening'],
    json_data = json.load(json_file)
    dictionary = json_data

POS_corpus = []

#dictionary의 키값으로 for문
for num in dictionary.keys() :
  #이중 리스트 안에 값으로 for문 
  for list_data in dictionary[num] :
    POS_corpus.append(komoran.get_plain_text(list_data[1]).split())

POS = []
for list in POS_corpus :
  for word in list :
    POS.append(word)

POS_SET = set(POS)
POS_SET = sorted(POS_SET)

word2idx = {'<PAD>': 0, '<UNK>': 1}

idx = 2 
for word_0 in POS_SET :
  word2idx[word_0] = idx 
  idx += 1 

idx2word = {idx: word for word, idx in word2idx.items()}

embedding_lookup_table = nn.Embedding(num_embeddings = len(word2idx), embedding_dim = 128)

dictionary2 = {}
for num in dictionary.keys() :
  POS_corpus = []
  for list_data in dictionary[num] :
    split_list_idx = []
    split_list = komoran.get_plain_text(list_data[1]).split()
    for word in split_list :
      split_list_idx.append(word2idx[word])
    POS_corpus.append(split_list_idx)
  dictionary2[num] = POS_corpus

print(dictionary2["1"])

with open('CNN_Embed.txt','w') as f:
  for num in dictionary2.keys() :
    f.write("(대화,%s)"%num)
    f.write("\n")
    for list in dictionary2[num] :
      #for list in list_list :
      findidx = torch.tensor(list)
      found = embedding_lookup_table.weight[findidx] 
      f.write(str(found.size()))
      f.write("\n")
    f.write("\n")

