# 02_Time-Series-prediction (개인 프로젝트)
(데이터는 타사 내부 데이터라 공개가 어려운 점 참고부탁드립니다.)
---
## Task(시계열 주문량 예측 모델)
- 축산 총 주문kg(등심 / 사골 채끝 등) 시계열 예측 모델

## Feature
- 등심, 사골_dif, 안심, 우둔 잡뼈_dif, 채끝등심, 홍두깨, 총합계 ( 사골 / 잡뼈의 경우 정상성을 만족하지 않아 차분 진행)
- 공휴일 / 현재시점 7일 뒤 공휴일 유무
- 3일전 총합계 평균 / 최소 / 최대 /편차
- 7일전 총합계 평균 / 최소 / 최대 /편차
- 주문량이 폭주되는 month 유무(1월 / 9월 /11월 / 12월)

## Model
- LSTM + CatBoost 앙상블 모델 구성
- LSTM ( 지난 90일 데이터로 다음 30일 예측)
- CatBoost ( 지난 60일 데이터로 7일 예측)

## Performance
- 24년 4월 한달 예측 (MAE 508 , 평균 주문량 3600) 
