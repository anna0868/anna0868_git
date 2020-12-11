### README.md
# Category Classification Task_최승아

### 모델설명
- 가맹점 카드내역 데이터를 17가지의 주어진 카테고리로 분류하는 모델

### 분석과정
- 카테고리의 특성을 파악하기 위해 17가지 카테고리의 데이터 특성을 파악
![ner_jupyter_notebook](./데이터불균형확인.png)


- 엔티티를 토큰화할때 토큰의 길이가 엔티티 자체보다 길어지는 경우, 정확한 엔티티 추출이 안될 수 있음 (토크나이저의 한계)
  - 이러한 경우에 대해서는 제외하고 학습할 수도 있지만, 더 넓은 범위를 커버하기 위해 포함하는 것으로 결정
  - e.g.)  첫 회를 시작으로 <13일:DAT>까지 -> ('▁13', 'B-DAT') ('일까지', 'I-DAT') (조사등이 같이 추출됨)
- 반대로 토큰화한 길이가 엔티티 자체보다 작은 경우 'I-tag' 토큰으로 해결가능
- pretrained sentencepiece를 사용하기 때문에 사전 변경은 안됨 (이것과 별개로 sp 사전을 변경 하는 방법은 따로 찾아봐야함) 
- pytorch-crf 라이브러리가 multi-gpu에서 안됨
  - 추후 변경
- BERT가 LM기반이라 그런지 오타에도 어느정도 강건한 편인듯함
- 문장 길이에 따라 NER 결과가 달라짐
- 영어 데이터에 대해서는 학습이 안되서 잘 안됨
- 사전에 나오는 '▁' 토큰과 우리가 흔히 사용하는 underscore '_'는 다르므로 주의할 것
- B 태그의 NER과 I 태그의 NER이 다를 경우를 방지하기 위해 BERT+Bi(LSTM or GRU)+CRF 구조로도 테스트 해봄
  - 장점 
    - 엔티티 토큰의 길이가 긴 경우는 잘 잡아냄
    - B 태그의 NER과 I 태그의 NER이 다른 경우가 확실히 줄어듬
  - 단점
    - 모델 사이즈가 커진다는 것
    - B 태그의 위치를 잘 못잡는 경우가 발생함  <12일:DAT>로 잡아야되는걸 앞문장의 구두점을 포함해서 <. 12일:DAT>로 잡거나, <1.83%:PNT>으로 잡아야 되는걸 1.8<3%:PNT> 잡기도함
  - 느낀점
    - B 태그 위치를 잘못잡는것 때문에 쓰기가 약간 애매하다는 생각이 듬 (보완이 필요함)
    - 학습은 GRU가 LSTM 보다 1 epoch정도 더 빠르게 성능이 올라감
- If you want to apply it to other languages, you don't have to change the model architecture. Instead, you just change vocab, pretrained BERT(from huggingface), and training dataset.

### Dataset
- [NER Dataset from 한국해양대학교 자연언어처리 연구실](https://github.com/kmounlp/NER)

### NER tagset
- 총 8개의 태그가 있음
    - PER: 사람이름
    - LOC: 지명
    - ORG: 기관명
    - POH: 기타
    - DAT: 날짜
    - TIM: 시간
    - DUR: 기간
    - MNY: 통화
    - PNT: 비율
    - NOH: 기타 수량표현
- 개체의 범주 
    - 개체이름: 사람이름(PER), 지명(LOC), 기관명(ORG), 기타(POH)
    - 시간표현: 날짜(DAT), 시간(TIM), 기간 (DUR)
    - 수량표현: 통화(MNY), 비율(PNT), 기타 수량표현(NOH)

### Results
- Epoch: 12 (without early stopping)
- num of train: 23032, num of val: 931
- Training set: ```00002_NER.txt```, ..., ```EXOBRAIN_NE_CORPUS_007.txt``` (1,425 files)
- Validation set: ```EXOBRAIN_NE_CORPUS_009.txt```, ```EXOBRAIN_NE_CORPUS_010.txt``` (2 files)

- Classification Report
  - 대체적으로 DAT, PER, NOH, ORG, PNT 순으로 높음
  - POH, LOC등은 좀 낮은 편
  - validation set 기준, macro avg F1: 87.56
<img src="./assets/classifcation_report_12_epoch.png" width="50%">

- Confusion Matrix
  - POH를 ORG로 예측하는 경우가 있음 (기타를 기관으로 분류하는 거니 어느정도 그럴 수 있다고 생각)
  - ORG를 PER로 예측하는 경우도 좀 있음 (수정해야되는 케이스)
<img src="./assets/best-epoch-12-step-1000-acc-0.960-cm.png" width="80%">

- Training & Evaluation Accurcay & Loss Graph
<img src="./assets/ner_training_acc_loss.gif" width="80%">

- Benchmark (Devset F1 scroe )

|Model|MacroAvg F1 score|Epoch|Date|
|:------:|:------:|:---:|:---:|
|KoBERT|0.8554|12|191129|
|**KoBERT+CRF**|**0.8756**|12|191129|
|KoBERT+BiLSTM+CRF|0.8659|12|191129|



### Requirements
```bash
pip install torch torchvision
pip install pytorch_pretrained_bert>=0.4.0
pip install mxnet>=1.5.0
pip install gluonnlp>=0.6.0
pip install sentencepiece>=0.1.6
pip install git+https://github.com/kmkurn/pytorch-crf#egg=pytorch_crf
pip install transformers
pip install tb-nightly
pip install future
```

### Model File Link
- [BERT CRF model file with validation](https://drive.google.com/file/d/1ZkWeR0gXPrUrOHe-xt4Im_Z9AXWFXmk7/view?usp=sharing)
- [BERT CRF model file with training all dataset](https://drive.google.com/open?id=1FDLe3SUOVG7Xkh5mzstCWWTYZPtlOIK8)
- [BERT CRF, BERT_alone sharing folder (including BiLSTM, BiGRU)](https://drive.google.com/drive/folders/1C6EKVpN5q1nENX2teqKuj_HHDfJoN47x?usp=sharing)

### train
```bash
python train_bert_crf.py 
```

### inference
```bash
python inference.py 
```

### Visualization
![BERT_NER_viz](./assets/kobert_ner_11_layer_viz.gif)

### Future work
- ~~Validation pipeline~~
- NER tag probability
- RestfulAPI
- Knowledge Distillation
- Apex fp16 half-precision
- Refactoring, Refactoring, Refactoring

### Reference Repo
- [NLP implementation by aisolab](https://github.com/aisolab/nlp_implementation)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf/blob/8f3203a1f1d7984c87718bfe31853242670258db/docs/index.rst)
- [SKTBrain KoBERT](https://github.com/SKTBrain/KoBERT)
- [Finetuning configuration from huggingface](https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_multiple_choice.py)
- [BERT Attention Visualization](https://github.com/jessevig/bertviz)
