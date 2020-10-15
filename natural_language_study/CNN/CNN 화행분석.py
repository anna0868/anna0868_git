!pip install PyKomoran
import json 
import torch 
import torch.nn as nn
from PyKomoran import *

komoran = Komoran("EXP")

from sklearn.feature_extraction import DictVectorizer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, fbeta_score, f1_score
import numpy as np
import random

def set_seed() :
  random.seed(777)
  np.random.seed(777)
  torch.manual_seed(777)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(777)

set_seed()

label_list = ['opening', 'request', 'wh-question', 'yn-question', 'inform', 'affirm','ack','expressive']
label_map = {label:i for i, label in enumerate(label_list)}

## 2. 형태소 분석한 Train data의 발화를 이용해 word2idx 구축
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

## word embedding_lookup_table 구축
embedding_lookup_table = nn.Embedding(num_embeddings = len(word2idx), embedding_dim = 128)

dictionary2 = {}
for num in dictionary.keys() :
  POS_corpus = []
  for list_data in dictionary[num] :
    split_list_idx = []
    split_list = komoran.get_plain_text(list_data[1]).split()
    for word in split_list :
      split_list_idx.append(word2idx[word])
    split_list_idx.extend([0]*(50-len(split_list_idx)))
    POS_corpus.append(split_list_idx)
  dictionary2[num] = POS_corpus


train_list = []
for num in dictionary.keys() :
  for list_data in dictionary[num] :
    split_list_idx = []
    split_list = komoran.get_plain_text(list_data[1]).split()
    for word in split_list :
      split_list_idx.append(word2idx[word])
    split_list_idx.extend([0]*(50-len(split_list_idx)))
    train_list.append(split_list_idx)

## 2. 형태소 분석한 Train data의 발화를 이용해 word2idx 구축
with open("SpeechAct_te.json") as json_file2:
    # json_data = {'1': [['user', '아름아 잘 잤니?', 'opening'], ['system', '네, 잘 잤습니다.', 'opening'],
    json_data2 = json.load(json_file2)
    dictionary_te = json_data2

########################################################################
test_list = []
for num_te in dictionary_te.keys() :
  for list_data_te in dictionary_te[num_te] :
    split_list_idx_te = []
    split_list_te = komoran.get_plain_text(list_data_te[1]).split()
    for word_te in split_list_te :
      split_list_idx_te.append(word2idx.get(word_te,1))
    split_list_idx_te.extend([0]*(50-len(split_list_idx_te)))
    test_list.append(split_list_idx_te)

################################################################ 3. Test data의 label를 이용해 label2idx 구축
label_list_2 = []
label_list = ['opening', 'request', 'wh-question', 'yn-question', 'inform', 'affirm','ack','expressive']
label2idx = {label:i for i, label in enumerate(label_list)}

for numb in dictionary_te.keys() :
  for list_data_tt in dictionary_te[numb] :
    label_list_2.append(list_data_tt[2])

test_label_list = []

for label2 in label_list_2:
  test_label_list.append(label2idx[label2])

test_tensor = torch.tensor(test_list)
test_label_tensor = torch.tensor(test_label_list)

## 3. Test data의 label를 이용해 label2idx 구축
label_list_1 = []
label_list = ['opening', 'request', 'wh-question', 'yn-question', 'inform', 'affirm','ack','expressive']
label2idx = {label:i for i, label in enumerate(label_list)}

for numb in dictionary.keys() :
  for list_data in dictionary[numb] :
    label_list_1.append(list_data[2])

train_label_list = []

for label in label_list_1:
  train_label_list.append(label2idx[label])

#list 자료형의 데이터를 Tensor 자료형의 데이터로 변환
# input Shape = (발화의 수, tfidf_size)
# output Shape = (발화의 수, 1)
train_tensor = torch.tensor(train_list)
train_label_tensor = torch.tensor(train_label_list)

epochs = 200
do = 0.7
learning_rate = 0.001


class CNN(torch.nn.Module):
  def __init__(self, vocab_size, num_labels):
    super(CNN, self).__init__()
    # embedding_lookup_table = nn.Embedding(num_embeddings = len(word2idx), embedding_dim = 128)
    self.word_embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=128, padding_idx=0)

    self.conv1 = torch.nn.Conv1d(128,128,2,1)
    self.conv2 = torch.nn.Conv1d(128,128,3,1)
    self.conv3 = torch.nn.Conv1d(128,128,4,1)
    #num_filter_sizes = 1 
    #num_filters = 49

    self.dropout = torch.nn.Dropout(do)
    self.fc1 = torch.nn.Linear(144, num_labels, bias = True)
    #self.fc2 = torch.nn.Linear(48, num_labels, bias = True)
    #self.fc3 = torch.nn.Linear(47, num_labels, bias = True)
    
  def forward(self, inputs) :
    #inputs shape = (2,50)
    # embedded.shape = (2,128,50)
    embedded = self.word_embed(inputs).permute(0,2,1)

    x1 = torch.max(self.conv1(embedded),1)
    x2 = torch.max(self.conv2(embedded),1)
    x3 = torch.max(self.conv3(embedded),1)
    
    x = torch.cat((x1[0],x2[0],x3[0]),1)

    y_pred = self.fc1(self.dropout(x))
    
    return y_pred

#GPU가 사용가능한지 확인하여 CPU / GPU 선택
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#train_tfidf_tensor shape = (발화의 수, tfidf_size)
model = CNN(1277, 8)
#model을 GPU/CPU로 이동
model.to(device)

#Optimizer 및 손실 함수 선언
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#TensorDataset을 이용하여 input/output data를 하나로 묶음
Train_dataset = torch.utils.data.TensorDataset(train_tensor, train_label_tensor)
#DataLoader 를 선언하여, batch size 만큼 데이터를 가져와서 모델 학습
train_DataLoader = torch.utils.data.DataLoader(Train_dataset, shuffle=True, batch_size=2)

# Train
model.train(True)
model.zero_grad()
for epoch in range(epochs):
  epoch_loss = 0
  for batch in train_DataLoader :
    #batch : (tfidf_data, label)
    batch = tuple(t.to(device) for t in batch)
    y_pred =  model(batch[0])

    loss = criterion(y_pred, batch[1])
    epoch_loss += loss.item()

    loss.backward()
    optimizer.step()
    model.zero_grad()
  if (epoch+1) % 10 == 0 :
    print(epoch, epoch_loss)

model.train(False)

Test_dataset = torch.utils.data.TensorDataset(test_tensor, test_label_tensor)
test_DataLoader = torch.utils.data.DataLoader(Test_dataset, shuffle=False, batch_size=1)

#Test
model.eval()
pred = None
label = None
for batch in test_DataLoader :
  #batch : (tfidf_data, label)
  batch = tuple(t.to(device) for t in batch)

  #gradient를 계산하지 않도록 선언
  with torch.no_grad():
    y_pred = model(batch[0])

  if pred is None:
    pred = y_pred.detach().cpu().numpy()
    label = batch[1].detach().cpu().numpy()
  else :
    pred = np.append(pred, y_pred.detach().cpu().numpy(), axis=0)
    label = np.append(label, batch[1].detach().cpu().numpy(), axis=0)

pred = np.argmax(pred, axis=1)

def percentage(data):
  k = int(1000000 * data)
  data1 = str(k/10000)
  return data1 + "%"

with open('CNN_EXPERIMENT.txt','w') as f:

  f.write('epochs : {0}\n'.format(epochs))
  f.write('dropout : {0}\n'.format(do))
  f.write('learning_rate : {0}\n'.format(learning_rate))
  f.write("\n")
  f.write('Macro average precision : ')
  f.write(percentage(precision_score(test_label_tensor, pred, average='macro')))
  f.write("\n")
  f.write('Micro average precision : ')
  f.write(percentage(precision_score(test_label_tensor, pred, average='micro')))
  f.write("\n\n")
  f.write('Macro average recall : ')
  f.write(percentage(recall_score(test_label_tensor, pred, average='macro')))
  f.write("\n")
  f.write('Micro average recall : ')
  f.write(percentage(recall_score(test_label_tensor, pred, average='micro')))
  f.write("\n\n")
  f.write('Macro average f1-score : ')
  f.write(percentage(f1_score(test_label_tensor, pred, average='macro')))
  f.write("\n")
  f.write('Micro average f1-score : ')
  f.write(percentage(f1_score(test_label_tensor, pred, average='micro')))

