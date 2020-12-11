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
1. top-1, 3, 5, Accuracy : 각 class 별 정확도
2. f1 Score : 각 class 별 f1 score (macro)
3. latency : 모델 실행 속도, n sample(s) / ms
