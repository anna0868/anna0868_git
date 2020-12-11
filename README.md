# Category Classification Task

### 모델설명
- 가맹점 카드내역 데이터를 17가지의 주어진 카테고리로 분류하는 모델

### 분석과정
- 카테고리 파악
	- 카테고리별 데이터 특성을 파악
	
	![데이터특징확인](./데이터특징확인.png)
	
	- 카테고리별 데이터 불균형 확인
	
	![데이터불균형확인](./데이터불균형확인.png)

### Metrics
1. 모델별 top-1, 3, 5, Accuracy : 각 class 별 정확도

	1-1.KoBERT
	|Class|top-1 Accuracy|top-3 Accuracy|top-5 Accuracy|
	|:------:|:------:|:------:|:------:|
	| 1|0.9104|0.9756|0.9834|
	| 2|0.5|0.625|0.625|
	| 3|0.9051|0.9379|0.9452|
	| 4|0.8675|0.9406|0.9589|
	| 5|0.9666|0.9877|0.9922|
	| 6|0.9295|0.9515|0.9515|
	| 7|0.8369|0.9239|0.9673|
	| 8|0.8602|0.9264|0.9595|
	| 9|0.8333|0.8666|0.9|
	|10|0.8990|0.9082|0.9449|
	|11|0.9509|0.9656|0.9656|
	|12|0.9782|0.9891|0.9891|
	|13|0.9689|0.9758|0.9827|
	|14|0.9797|0.9932|0.9932|
	|15|0.85|0.9|0.9|
	|16|0.7272|0.8181|0.8181|
	|17|0.95|0.95|0.95|	


	1-2.KoElectra
	|Class|top-1 Accuracy|top-3 Accuracy|top-5 Accuracy|
	|:------:|:------:|:------:|:------:|
	| 1|0.8685|0.9679|0.9845|
	| 2|0.25|0.5|0.75|
	| 3|0.8686|0.9124|0.9416|
	| 4|0.7488|0.8310|0.8949|
	| 5|0.9337|0.9760|0.9888|
	| 6|0.9030|0.9383|0.9603|
	| 7|0.7065|0.8913|0.8913|
	| 8|0.8235|0.8676|0.9191|
	| 9|0.75|0.8166|0.8666|
	|10|0.7431|0.7889|0.8715|
	|11|0.8676|0.9068|0.9215|
	|12|0.9565|0.9673|0.9782|
	|13|0.9103|0.9344|0.9517|
	|14|0.9797|0.9932|0.9932|
	|15|0.7|0.9|0.9|
	|16|0.6363|0.7272|0.7272|
	|17|0.9|0.95|0.95|


	1-3.RNN
	|Class|top-1 Accuracy|top-3 Accuracy|top-5 Accuracy|
	|:------:|:------:|:------:|:------:|
	| 1|0.8110|0.9580|0.9723|
	| 2|0.125|0.5|0.5|
	| 3|0.7883|0.8905|0.9124|
	| 4|0.6849|0.8082|0.9680|
	| 5|0.9360|0.9872|0.9966|
	| 6|0.8854|0.9162|0.9162|
	| 7|0.7282|0.7934|0.8369|
	| 8|0.7683|0.8198|0.9595|
	| 9|0.6	  |0.6833|0.7333|
	|10|0.8165|0.8348|0.8623|
	|11|0.8480|0.8725|0.8921|
	|12|0.9782|0.9782|0.9782|
	|13|0.8896|0.9137|0.9206|
	|14|0.9662|0.9729|0.9729|
	|15|0.8|0.9|0.9|
	|16|0.4545|0.6363|0.7272|
	|17|0.85|0.85|0.85|


	1-4.CNN
	|Class|top-1 Accuracy|top-3 Accuracy|top-5 Accuracy|
	|:------:|:------:|:------:|:------:|
	| 1|0.7723|0.979 |0.9867|
	| 2|0.0|0.0|0.125|
	| 3|0.6496|0.8467|0.9708|
	| 4|0.4840|0.7397|0.8767|
	| 5|0.9449|0.9838|0.9894|
	| 6|0.8061|0.8766|0.9118|
	| 7|0.2717|0.6086|0.8043|
	| 8|0.7352|0.7977|0.8492|
	| 9|0.3|0.6|0.6666|
	|10|0.4587|0.7064|0.7614|
	|11|0.7598|0.8823|0.8921|
	|12|0.9130|0.9456|0.9673|
	|13|0.8827|0.9206|0.9310|
	|14|0.9324|0.9391|0.9527|
	|15|0.35|0.35|0.55|
	|16|0.0|0.0909|0.4545|
	|17|0.45|0.8|0.8|


2. f1 Score : 각 class 별 f1 score (macro)
	
	2-1.KoBERT
	|Class|f1 Score|
	|:------:|:------:|
	| 1|0.9191|
	| 2|0.5333|
	| 3|0.9236|
	| 4|0.8407|
	| 5|0.9660|
	| 6|0.9154|
	| 7|0.7897|
	| 8|0.8897|
	| 9|0.7936|
	|10|0.8909|
	|11|0.9440|
	|12|0.9782|
	|13|0.9706|
	|14|0.9764|
	|15|0.8947|
	|16|0.6153|
	|17|0.9500|
	
	2-2.KoElectra
	|Class|f1 Score|
	|:------:|:------:|
	| 1|0.8713|
	| 2|0.4|	
	| 3|0.8880|
	| 4|0.7865|
	| 5|0.9214|
	| 6|0.9050|
	| 7|0.7514|
	| 8|0.8565|
	| 9|0.7142|
	|10|0.6982|
	|11|0.8634|
	|12|0.9462|
	|13|0.9041|
	|14|0.9570|
	|15|0.8|
	|16|0.6086|
	|17|0.9230|
	
	2-3.RNN
	|Class|f1 Score|
	|:------:|:------:|
	| 1|0.8412|
	| 2|0.1428|
	| 3|0.8388|
	| 4|0.7228|
	| 5|0.8904|
	| 6|0.9054|
	| 7|0.7282|
	| 8|0.8085|
	| 9|0.6666|
	|10|0.8054|
	|11|0.8606|
	|12|0.9326|
	|13|0.9132|
	|14|0.9828|
	|15|0.8205|
	|16|0.3846|
	|17|0.8717|
	
	2-4.CNN
	|Class|f1 Score|
	|:------:|:------:|
	| 1|0.7753|
	| 2|0.0|
	| 3|0.6912|
	| 4|0.5353|
	| 5|0.8685|
	| 6|0.7689 
	| 7|0.3846|
	| 8|0.8113|
	| 9|0.4337|
	|10|0.5376|
	|11|0.775|
	|12|0.9385|
	|13|0.9126|
	|14|0.9616|
	|15|0.4999|
	|16|0.0|
	|17|0.6|

3. latency : 모델 실행 속도, n sample(s) / ms
	|Model|latency|
	|:------:|:------:|
	|KoBERT|1.07 samples/ms|
	|KoElectra|0.90 samples/ms|
	|RNN|0.21 samples/ms|
	|CNN|1.26 samples/ms|
*latency : 평가 시 총 걸린 시간을 구해 test data의 갯수로 나누어 sample당 걸린시간(ms)을 측정하였다. 

### 결론
각 모델별 f1 score(macro) 는 KoBERT 모델이 약 87%로 가장 높았고,  Electra 모델이 약 81%, RNN 모델 약 77%, CNN 모델 약 60%로 CNN이 가장 낮았다.
각 모델별 속도는 CNN이 약 1.3 samples / ms 로 가장 빨랐고, KoBERT 모델이 약 1.1 samples / ms , KoElectra 모델이 0.9 samples / ms , RNN 모델이 0.2 samples / ms 로 RNN이 가장 느렸다.

Score와 모델 속도를 동시에 고려하면 현재까지 분석한 바에 있어 KoBERT 모델이 가장 좋은 모델로 판단된다. 
클래스별 데이터가 불균등하기 떄문에 각 클래스별로 Accuracy, f1 score가 크게 차이가 났다. 데이터 불균등을 보완하기 위해 공개된 가맹점 데이터(경기도 성남시, 수원시의 지역화폐 가맹점 현황)을 KoBERT, KoElectra를 활용해 생성한 모델로 라벨링하여 다시 학습 데이터로 추가해보기도 하였으나 성능이 크게 향상되지는 않았다. 또한, SMOTE, under sampling 등도 적용해 보았지만 오히려 성능이 떨어졌다. 따라서 현재의 모델에서 성능을 높이기 위해서는 데이터 불균등을 해결하기 위한 방법에 대해 좀 더 연구가 필요할 것이다.
