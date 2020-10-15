!pip install PyKomoran
import json 
import torch
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

tfidfvect = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')

train_tfidf_list = []
train_label_list = []
test_tfidf_list = []
test_label_list = []

with open("SpeechAct_tr.json") as json_file:
    # json_data = {'1': [['user', '아름아 잘 잤니?', 'opening'], ['system', '네, 잘 잤습니다.', 'opening'],
    json_data = json.load(json_file)
    dictionary = json_data

POS_corpus = []
POS_label = []

#dictionary의 키값으로 for문
for num in dictionary.keys() :
  #이중 리스트 안에 값으로 for문 
  for list_data in dictionary[num] :
    POS_corpus.append(' '.join(komoran.get_morphes_by_tags(list_data[1], tag_list=['NNP','NNG','VV'])))
    POS_label.append(list_data[2])

for i in range(len(POS_label)):
  train_label_list.append(label_map[POS_label[i]])


tfidfvect.fit(POS_corpus)
train_tfidf_list = tfidfvect.transform(POS_corpus).toarray().tolist()


with open("SpeechAct_te.json") as json_file_test:
    # json_data = {'1': [['user', '아름아 잘 잤니?', 'opening'], ['system', '네, 잘 잤습니다.', 'opening'],
    json_data_test = json.load(json_file_test)
    dictionary_test = json_data_test

POS_corpus_test = []
POS_label_test = []

#dictionary의 키값으로 for문
for num_test in dictionary_test.keys() :
  #이중 리스트 안에 값으로 for문 
  for list_data_test in dictionary_test[num_test] :
    POS_corpus_test.append(' '.join(komoran.get_morphes_by_tags(list_data_test[1], tag_list=['NNP','NNG','VV'])))
    POS_label_test.append(list_data_test[2])

for i in range(len(POS_label_test)):
  test_label_list.append(label_map[POS_label_test[i]])

tfidfvect.fit(POS_corpus)
test_tfidf_list = tfidfvect.transform(POS_corpus_test).toarray().tolist()

#list 자료형의 데이터를 Tensor 자료형의 데이터로 변환
# input Shape = (발화의 수, tfidf_size)
# output Shape = (발화의 수, 1)
train_tfidf_tensor = torch.tensor(train_tfidf_list)
train_label_tensor = torch.tensor(train_label_list)
test_tfidf_tensor = torch.tensor(test_tfidf_list)
test_label_tensor = torch.tensor(test_label_list)

# DEVICE 선언 및 모델 선언
class Perceptron(torch.nn.Module):
  def __init__(self, tfidf_size, num_label):
    super(Perceptron, self).__init__()
    self.linear = torch.nn.Sequential( torch.nn.Linear(tfidf_size, 100), torch.nn.Tanh(), torch.nn.Linear(100, num_label),)

  def forward(self, tfidf_input):
    y_pred = self.linear(tfidf_input)

    return y_pred

#GPU가 사용가능한지 확인하여 CPU / GPU 선택
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#train_tfidf_tensor shape = (발화의 수, tfidf_size)
model = Perceptron(tfidf_size = train_tfidf_tensor.shape[1], num_label = len(label_list))
#model을 GPU/CPU로 이동
model.to(device)

#Optimizer 및 손실 함수 선언
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#TensorDataset을 이용하여 input/output data를 하나로 묶음
Train_dataset = torch.utils.data.TensorDataset(train_tfidf_tensor, train_label_tensor)
Test_dataset = torch.utils.data.TensorDataset(test_tfidf_tensor, test_label_tensor)

#DataLoader 를 선언하여, batch size 만큼 데이터를 가져와서 모델 학습
#shuffle 여부 결정
train_DataLoader = torch.utils.data.DataLoader(Train_dataset, shuffle=True, batch_size=4)
test_DataLoader = torch.utils.data.DataLoader(Test_dataset, shuffle=False, batch_size=1)

# Train
model.train(True)
model.zero_grad()
for epoch in range(200):
  epoch_loss = 0
  for batch in train_DataLoader :
    #batch : (tfidf_data, label)
    batch = tuple(t.to(device) for t in batch)
    y_pred = model(batch[0])

    loss = criterion(y_pred, batch[1])
    epoch_loss += loss.item()

    loss.backward()
    optimizer.step()
    model.zero_grad()
  if (epoch+1) % 10 == 0 :
    print(epoch, epoch_loss)

model.train(False)

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
  data1 = str(100 * data)
  return data1 + "%"

with open('MLP.txt','w') as f:


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

