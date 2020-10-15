!pip install PyKomoran
import json 
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

with open('CNN_w2ii2w.txt','w') as f:
  f.write("word2idx")
  f.write("\n")
  for word in word2idx.keys() :
    f.write("%s\t"%word)
    f.write("%s\n"%word2idx[word])
  f.write("\n")
  f.write("idx2word")
  f.write("\n")
  for idx in idx2word.keys() :
    f.write("%s\t"%idx)
    f.write("%s\n"%idx2word[idx])

