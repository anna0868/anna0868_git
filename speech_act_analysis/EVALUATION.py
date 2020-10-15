!pip install PyKomoran

from PyKomoran import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

komoran = Komoran("EXP")

from google.colab import files
uploaded = files.upload()

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, fbeta_score, f1_score

import json

with open("Evaluation.json") as json_file:
    json_data = json.load(json_file)

list_b = []
for data in list(json_data.keys()) :
    json_array = []
    json_array = json_data[data]
    list_b.append(json_array)  

def percentage(data):
  data1 = str(100 * data)
  return data1 + "%"

#예측값
y_pred = list_b[0]
#실제정답
y_true = list_b[1]

with open('EVAL.txt','w') as f:

  f.write("Confusion matrix")
  con_mat = confusion_matrix(y_true, y_pred)
  f.write("\n")
  for i in range(3):
    for k in range(3):
      f.write("%2d\t"%con_mat[i][k])
    f.write("\n")

  f.write("\n")
  f.write("Accuracy : ")
  f.write(percentage(accuracy_score(y_true, y_pred)))
  f.write("\n\n")
  f.write('Macro average precision : ')
  f.write(percentage(precision_score(y_true, y_pred, average='macro')))
  f.write("\n")
  f.write('Micro average precision : ')
  f.write(percentage(precision_score(y_true, y_pred, average='micro')))
  f.write("\n\n")
  f.write('Macro average recall : ')
  f.write(percentage(recall_score(y_true, y_pred, average='macro')))
  f.write("\n")
  f.write('Micro average recall : ')
  f.write(percentage(recall_score(y_true, y_pred, average='micro')))
  f.write("\n\n")
  f.write('Macro average f1-score : ')
  f.write(percentage(f1_score(y_true, y_pred, average='macro')))
  f.write("\n")
  f.write('Micro average f1-score : ')
  f.write(percentage(f1_score(y_true, y_pred, average='micro')))

