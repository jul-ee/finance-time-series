# 📋 Finance Time Series Classification

본 프로젝트는 이더리움(ETH)의 분당 시계열 데이터를 기반으로 가격 변화 패턴을 분석하고, 라벨링 및 특징 추출 기법을 적용하여 가격 방향성 분류 모델을 구축하는 것을 목표로 합니다. 

금융 시계열 데이터의 복잡성을 고려하여 정보 누수를 방지하고 효과적인 특징을 설계하는 데 중점을 두었습니다.

> 🛠️ **Tech Stack**
> 
>Language: &nbsp;Python  
Data Analysis & EDA: &nbsp;pandas, numpy, tqdm, Jupyter Notebook  
Feature Engineering: &nbsp;ta, shap, sklearn.feature_selection  
Machine Learning:<br>- Modeling: &nbsp;`scikit-learn` (LogisticRegression, RandomForest, Bagging, Stacking), `XGBoost`, `LightGBM`<br>- Evaluation: &nbsp;`scikit-learn.metrics`

<br>
<br>

## 프로젝트 목표

- 시계열 데이터 기반 가격 방향성 라벨링 설계
- 기술적 지표 및 통계적 피처를 활용한 특징 추출
- 정보 누수 방지 기반 교차검증 적용
- Bagging 및 Stacking 등 앙상블 모델을 활용한 성능 비교 및 예측

<br>
<br>

## 프로젝트 개요

- Data: &nbsp;이더리움(ETH) 분 단위 시세 (CSV)
- Labeling:
    - Price Change Direction, Using Moving Average, Local Min-Max, Trend Scanning
- 검증 기법: &nbsp;PurgedKFold 기반 시계열 적합 교차검증
- 성능 평가 지표: &nbsp;Accuracy, Precision, Recall, AUC

<br>
<br>

## 전체 분석 프로세스

```
📁 finance-time-series-project
│
└── 📂 1. 데이터 로드 및 초기 탐색
    ├──  ETH 분당 시계열 CSV 및 라이브러리 불러오기
    ├──  시계열 인덱스 정렬
    └──  종가 시계열 흐름 시각화

    📂 2. Data Labeling
    ├── 📁 2-1. Price Change Direction
    │   ├──  10분 지연 모멘텀 신호 생성 (np.sign +1)
    │   └──  모멘텀 신호 상위 15개 확인
    │
    ├── 📁 2-2. Using Moving Average
    │   └──  이동평균 기반 신호 민감도 조정
    │
    ├── 📁 2-3. Local Min-Max
    │   ├──  get_local_min_max() 함수 정의 (wait=3 조건)
    │   ├──  극값 시각화: 전체 및 확대
    │   └──  min/max 기반 trend 라벨링
    │
    └── 📁 2-4. Trend Scanning
        ├──  선형 회귀 기반 t-value 계산 함수 정의
        ├──   각 시점별 회귀 적용 후 유의미한 t_val, bin 도출
        └──  label DataFrame 상위 20개 확인

    📂 3. Feature Engineering
    ├── 📁 3-1. 환경 구성 및 데이터 슬라이싱
    │   ├──  ta, shap 설치 및 관련 라이브러리 import
    │   └──  MDI 방식 피처 중요도 함수 정의
    │
    ├── 📁 3-2. Technical Index 적용
    │   ├──  기술적 분석 지표 Feature 생성
    │   └──  Volume, Volatility, Trend 기반
    │
    ├── 📁 3-3. 수익률/변동성 지표 적용
    │   ├──  수익률 및 변동성 기반 파생 피처 생성
    │   └──  ret, std, vol_change 기반
    │
    └── 📁 3-4. 피처 중요도 분석 및 선택
        ├──  X, y 분리 및 StandardScaler 정규화
        ├──  각 시점별 회귀 적용 후 유의미한 t_val, bin 도출
        └──  label DataFrame 상위 20개 확인

    📂 4. Feature Selection methods
    ├──  MDI: Mean Decrease Impurity
    ├──  MDA: Mean Decrease Accuracy
    ├──  RFE CV: Recursive Feature Elimination
    ├──  SFS: Sequential Feature Selection
    └──  SHAP: Shapley Additive explanations
         └──  클래스별 summary plot 시각화
			   └──  전체 피처 중요도 정렬 → 모델 해석
    
    📂 5. 학습 데이터 저장 및 유틸리티 모듈 구성
    ├──  최종 feature + label 포함된 데이터 저장 (.pkl)
    └──  ml_get_train_times1, PKFold 클래스 정의 (정보 누수 방지용)

    📂 6. 모델 학습 및 평가
    ├── 📁 6-1. 데이터 로딩 및 준비
    │   ├──  .pkl 데이터 로딩
    │   ├──  t_value 이진화
    │   └──  클래스 비율 확인
    │
    ├── 📁 6-2. 데이터 분할 및 정규화
    │   ├──  train/test 비율 설정 (70/20)
    │   ├──  X, y 분리 및 정규화
    │   └──  train/test 셋 구성 및 학습용 1000개로 제한
    │
    ├── 📁 6-3. 교차검증 설정
    │   ├──  PKFold 적용 (n_splits=4)
    │   └──  n_splits 파라미터 변경 실험
    │
    ├── 📁 6-4. 베이스라인 학습 및 하이퍼파라미터 튜닝
    │   ├──  BaggingClassifier(RandomForest) 구성
    │   └──  GridSearchCV로 하이퍼파라미터 튜닝
    │
    ├── 📁 6-5. 최종 학습 및 예측
    │   ├──  best estimator로 전체 학습 데이터 재학습
    │   ├──  predict, predict_proba 수행
    │   ├──  성능 지표 출력
    │   └──  ROC Curve 시각화 및 AUC 해석
    │
    └── 📁 6-6. 추가 모델 학습 및 성능 비교
        ├──  개별 모델(XGBoost, LightGBM) 학습 및 성능 비교
        ├──  Ensemble Stacking
        │     └──  RF + XGB + LGBM → LogisticRegression
        ├──  Threshold 최적화 기반 성능 향상
        │     └──  Precision-Recall Curve 기반 F1 최적 지점 선택
        ├──  스태킹 모델 최종 성능
        │     ├──  Accuracy: 0.8010
        │     ├──  Precision: 0.5955
        │     ├──  Recall: 0.9636
        │     └──  AUC: 0.9118
        ├──  Confusion Matrix 시각화 및 해석
        ├──  ROC Curve (probability 기반) 시각화
        │
        └──  결과 종합 비교 및 최종 모델 선정 사유
             • 개별 모델 대비 전반적인 성능 균형 우수
             • 민감한 이벤트 탐지에 강한 재현율 확보
             • AUC 기준으로도 안정적인 분류 가능성 확보
             → 최종 모델로 StackingClassifier 선정
```


<br>

## 결론

개별 분류 모델(RandomForest, XGBoost, LightGBM)을 적용해보았으나, 하이퍼파라미터 튜닝에도 불구하고 성능 상의 한계을 확인하였다. 단일 모델만으로는 복잡한 패턴을 충분히 반영하기 어렵다고 판단하였다.

이에 따라 다양한 모델의 예측 결과를 통합해 성능을 향상시키기 위한 스태킹 앙상블 모델(Ensemble Stacking)을 도입하였다.

- Base model:  RandomForest, XGBoost, LightGBM
- Meta model:  Logistic Regression.

최적의 Threshold까지 적용한 결과, 성능이 유의미하게 향상된 것을 확인하였다.

|           | Accuracy | Precision | Recall | AUC    |
| --------- | -------- | --------- | ------ | ------ |
| **Score** | 0.8010   | 0.5955    | 0.9636 | 0.9118 |



개별 모델 대비 전반적인 성능과 재현율 확보를 통해 가격 변동 감지에 민감하게 반응하며, AUC 기준으로 안정적인 분류 가능성이 확보된 것으로 판단하였다.

최종 모델: `StackingClassifier (RF + XGB + LGBM → Logistic Regression)`


<br>
<br>
<br>


## 인사이트 및 회고

시계열 라벨링과 정보 누수 방지 전략이 예측력 향상에 중요하게 작용한다는 것을 확인하였다.

이동평균 민감도 조정, 극값 탐지, 회귀 기반 t-value 라벨링 방법을 선택할 때 예측 성능과 해석 가능성을 모두 고려해야 한다.

시계열을 분류 문제에서도 threshold 최적화가 성능 향상에 효과적일 수 있다는 것을 확인하였다.

다양한 Feature Selection 기법과 앙상블 모델을 비교하면서 데이터와 목적에 맞는 분석 방향을 잡아가는 연습이 되었다.


<br>

>본 프로젝트는 시계열 데이터의 특성에 대한 이해를 기반으로 모델의 예측 성능과 해석력을 함께 고려한 분석에 중점을 두었습니다.

