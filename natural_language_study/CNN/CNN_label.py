import json

with open("SpeechAct_tr.json") as json_file:
    json_data = json.load(json_file)
    dictionary = json_data

label_list_1 = []
label_list = ['opening', 'request', 'wh-question', 'yn-question', 'inform', 'affirm','ack','expressive']
label2idx = {label:i for i, label in enumerate(label_list)}

for num in dictionary.keys() :
  for list_data in dictionary[num] :
    label_list_1.append(list_data[2])

train_label_list = []

for label in label_list_1:
  train_label_list.append(label2idx[label])


with open("SpeechAct_te.json") as json_file2:
    json_data2 = json.load(json_file2)
    dictionary2 = json_data2

label_list_2 = []

for num2 in dictionary2.keys() :
  for list_data2 in dictionary2[num2] :
    label_list_2.append(list_data2[2])

test_label_list = []

for label2 in label_list_2:
  test_label_list.append(label2idx[label2])

train_len = str(len(train_label_list))
test_len = str(len(test_label_list))

 
with open('CNN_label.txt','w') as f:
  f.write('Train_label_list_길이 : ')
  f.write(train_len)
  f.write("\n")
  f.write('Train_Data_100번째_인덱스_label : ')
  f.write(str(train_label_list[99]))
  f.write("\n")
  f.write('Test_label_list_길이 : ')
  f.write(test_len)
  f.write("\n")
  f.write('Test_Data_100번째_인덱스_label : ')
  f.write(str(test_label_list[99]))
  f.write("\n")

