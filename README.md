# SKN07-2nd-4Team 

> **팀원 : 김성근, 대성원, 유수현, 윤정연, 정승연**
</br>

<div align="center">
  <h2><strong> 📞 통신사 이용 고객 이탈 예측 📞 </h2></strog>
  2024.12.16 ~ 2024.12.17 
</div>
<br><br>
<div align="center">
    <div>
        <img src="https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?style=flat&logo=Visual%20Studio%20Code&logoColor=white"/>
        <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>
        <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=flat&logo=Artificial%20Intelligence&logoColor=white"/>
        <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white"/>
        <img src="https://img.shields.io/badge/Deep%20Learning-FCFC00?style=flat&logo=Artificial%20Intelligence&logoColor=white"/>
        <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white">
        <br/>
        <img src="https://img.shields.io/badge/Git-F05032?style=flat&logo=Git&logoColor=white"/>
        <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=GitHub&logoColor=white"/>
        <img src="https://img.shields.io/badge/Discord-5865F2?style=flat&logo=Discord&logoColor=white"/>
    </div>
</div>
<br><br>
    
---
<br>

### 1. 프로젝트 개요 
이 프로젝트는 통신사의 고객 데이터를 기반으로, 고객의 이탈(Churn)을 예측하는 모델을 개발하는 것입니다.
고객 이탈 예측은 비즈니스의 핵심 문제 중 하나로, 고객을 유지하는 것이 수익성에 큰 영향을 미칩니다.
고객이 이탈할 가능성을 예측하는 모델을 구축하여, 통신사는 이탈 가능성이 높은 고객을 사전에 식별하고, 그에 맞는 마케팅 전략을 통해 고객을 유지할 수 있게 됩니다.

</br>

### 2. 프로젝트 배경 
- 고객 이탈 문제
통신사와 같은 서비스 기반 기업에게 고객 이탈은 중요한 경영 문제입니다. </br>
고객이 서비스를 중단하거나 경쟁사의 서비스로 이동하면, 기업은 수익 감소는 물론, 새로운 고객을 유치하기 위한 마케팅 비용이 급증하게 됩니다. </br>
고객 이탈을 예측하는 능력을 강화하면, 기업은 더욱 효과적으로 고객을 유지할 수 있으며, 이를 통해 고객 충성도를 높이고 수익성을 개선할 수 있습니다.

- 고객 이탈 데이터 분석의 필요성
과거의 고객 데이터를 분석하여 고객의 이탈 패턴을 파악하는 것이 필요합니다.</br>
고객의 나이, 사용 기간, 서비스 이용 패턴, 요금제 등 다양한 특성을 바탕으로 이탈 예측 모델을 개발할 수 있습니다.</br>
데이터 분석을 통해 고객의 행동을 이해하고, 이탈을 방지할 수 있는 전략을 세울 수 있으며, 맞춤형 고객 관리가 가능해집니다.
</br>

### 3. 프로젝트 목표 

고객 이탈 예측 모델 개발</br>
- 고객의 이탈 가능성을 예측할 수 있는 머신러닝 모델을 개발합니다.

활용 목표
- 모델을 통해 이탈 가능성이 높은 고객을 식별한 후, 고객 세분화를 통해 다양한 맞춤형 대응 및 맞춤형 마케팅 전략을 제시합니다.</br>
  예를 들어, 특정 연령대나 특정 요금제에서 이탈이 많다면 해당 그룹에 대한 특별한 혜택을 제공할 수 있습니다.
- 예측 모델을 통해 고객 이탈을 줄이고, 고객 유지율을 향상시키는 전략을 제시합니다.


</br>

### 4. 프로젝트 과정
 (1) Dataset 준비
 > Telecom Churn : 13.37MB
 <br>dtypes: float64(26), int64(9), object(23)
 <br>RangeIndex: 51047 entries, 0 to 51046
 <br>Data columns : total 58 columns
 <br>Target : Yes(1) : No(0) = 1 : 4
 <br>출처 : Kaggle (https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom/data?select=cell2celltrain.csv)

<br><br>

 (2) EDA
 >58개의 Column을 6개의 DataFrame으로 분리하여 Heatmap 시각화
 <br>
 <img width="400px" src="image/billing_usage_heatmap.png" />
 <img width="400px" src="image/call_stats_heatmap.png" /> 
 <img width="400px" src="image/customer_churn_heatmap.png" /> 
 <img width="400px" src="image/customer_lifestyle_heatmap.png" /> 
 <img width="400px" src="image/customer_profile_heatmap.png" /> 
 <img width="400px" src="image/handset_details_heatmap.png" /> 
 <br>
 
<br><br>

 >Feature 중요도 순위에 따라 상위 40개 항목을 추출하여 훈련 데이터로 사용
 <img width="900px" height="700" src="image/feature_importance.png" />

<br><br>

 >상위 40개 항목 Column 정보 확인
 <img width="500px" height="1000" src="image/columns.png" />

<br><br>
 
 (3) 데이터 전처리 
 
▶ 결측치 제거 
``` python
df_all = train_file.dropna().copy()
df_all.isnull().sum()
```

<img width="250" alt="image" src="https://github.com/user-attachments/assets/1517f21a-2fb1-407d-92e8-a906499c8d39" /> <img width="250" alt="image" src="https://github.com/user-attachments/assets/487f1ed0-9284-48ba-9739-59bf24bbd03f" />
<br><br>

  ▶ 라벨값(Churn) object --> int 변경 
``` python
churn_label = {'No': 0.0, 'Yes': 1.0} # 유지 0, 이탈 1
df_all['Churn'] = df_all['Churn'].map(churn_label)
df_all
```

<br><br>
  ▶ 라벨값(Churn) 비율 확인
``` python
import numpy as np
np.unique(df_all['Churn'], return_counts=True)
```
<img width="600" alt="image" src="https://github.com/user-attachments/assets/e7e47fef-2fb2-47bc-aefc-48eaae314849" />

<br><br>
  ▶ 모델 훈련에 사용할 컬럼만 선택하여 x_data,y_data 생성
``` python
 # 상관 계수가 높은 컬럼을 선택
 X_data = df_all[['CustomerID','MonthlyRevenue','MonthlyMinutes','TotalRecurringCharge','OverageMinutes','RoamingCalls','PercChangeMinutes','PercChangeRevenues','MonthsInService','RetentionCalls','RetentionOffersAccepted','NewCellphoneUser','NotNewCellphoneUser','ReferralsMadeBySubscriber','AdjustmentsToCreditRating','MadeCallToRetentionTeam','CreditRating','PeakCallsInOut','OffPeakCallsInOut','ReceivedCalls','UnansweredCalls','OutboundCalls','DroppedBlockedCalls','DroppedCalls','InboundCalls','BlockedCalls','DirectorAssistedCalls','CustomerCareCalls','CallWaitingCalls','CurrentEquipmentDays','HandsetRefurbished','IncomeGroup','PrizmCode','Occupation','MaritalStatus','HandsetModels','AgeHH1','ChildrenInHH','HandsetPrice','ThreewayCalls','Handsets']].copy()

# 이탈율
y_data = df_all['Churn']
```

<br><br>
 ▶ One-Hot 인코딩 
``` python
    # one-hot 인코딩 
    if 'PrizmCode' in df_file.columns:
        df_file = pd.get_dummies(df_file, columns=['PrizmCode'],drop_first=False)
    if 'Occupation' in df_file.columns:
        df_file = pd.get_dummies(df_file, columns=['Occupation'],drop_first=False)
    if 'MaritalStatus' in df_file.columns:
        df_file = pd.get_dummies(df_file, columns=['MaritalStatus'],drop_first=False)
    
    # Unknown -> -1 , 나머지는 숫자로 형 변환
    def label_handset_price(value):
        if value == 'Unknown':
            return -1  # Unknown 값을 -1로 라벨링
            # return np.nan  # Unknown 값을 NaN으로 처리 (XGBoost사용)
        else:
            return int(value)  # 나머지는 숫자로 변환
    if df_file['HandsetPrice'].dtype == 'object':
        df_file['HandsetPrice'] = df_file['HandsetPrice'].apply(label_handset_price)

    return df_file
```

<br><br>

 (4) 모델링 
 
 ▶ 데이터셋 분리
``` python
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
```
<br>

 ▶ 모델 훈련
 ## Machine Learning

``` python
# 모델 정의
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=100),
    "SGDClassifier": SGDClassifier(loss='hinge'),
    "KNN": KNeighborsClassifier()
}

# Stratified K-Fold 설정
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# NumPy 배열로 변환 
X_resampled = X_resampled.to_numpy() if isinstance(X_resampled, pd.DataFrame) else X_resampled
y_resampled = y_resampled.to_numpy() if isinstance(y_resampled, pd.Series) else y_resampled

# 모델 훈련 및 평가
results = {}
for name, model in models.items():
    print(f"Model: {name}")
    fold_accuracies = []
    fold_roc_aucs = []

    for train_idx, val_idx in skf.split(X_resampled, y_resampled):
        if name in ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]:
            X_fold_train, X_fold_val = X_resampled[train_idx], X_resampled[val_idx]
        else:
            X_fold_train, X_fold_val = X_train_scaler[train_idx], X_train_scaler[val_idx]

        y_fold_train, y_fold_val = y_resampled[train_idx], y_resampled[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        y_preds = model.predict(X_fold_val)
        y_probs = model.predict_proba(X_fold_val)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_fold_val, y_preds)
        roc_auc = roc_auc_score(y_fold_val, y_probs) if y_probs is not None else None
        fold_accuracies.append(acc)
        if roc_auc is not None:
            fold_roc_aucs.append(roc_auc)
```
<br>

▶ 모델 성능 개선
``` python
# 튜닝을 위한 공통 부분 작성 (위와 중복되는 부분 있음)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 불균형 데이터 처리
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# NumPy 배열로 변환

# 교차검증 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 학습 및 평가 함수
def go_train(select_model,model_name):
    accuracies = []
    roc_aucs = []

    # 교차검증 
    for train_idx, val_idx in skf.split(X_resampled, y_resampled):
    
        # 교차검증용 훈련 세트와 검증 세트 분리

        # 모델 학습 
        select_model.fit(X_train, y_train)

        # 검증 세트 평가
        y_val_pred = select_model.predict(X_val)
        y_val_prob = select_model.predict_proba(X_val)[:, 1]
        accuracies.append(accuracy_score(y_val, y_val_pred))
        roc_aucs.append(roc_auc_score(y_val, y_val_prob))

    # 테스트 세트
    select_model.fit(X_resampled, y_resampled)  # 전체 훈련 데이터로 재학습
    y_test_pred = select_model.predict(X_test)
    y_test_prob = select_model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_prob)
```
</br>

### 5. 프로젝트 결과

</br>

**1) 모델 성능 평가**

</br>

각 모델의 성능은 교차 검증을 통해 검토되었으며, 주요 평가지표인 **정확도(Accuracy)**, **정밀도(Precision)**, **재현율(Recall)**, **F1 점수** 및 **ROC-AUC**를 기준으로 평가하였습니다.

</br>

**2) 모델링**

</br>

로지스틱 회귀(Logistic Regression), 결정 트리(Decision Tree), 랜덤 포레스트(Random Forest), 그래디언트 부스팅(Gradient Boosting), XGBoost 등 다양한 모델이 실험되었습니다.
- **로지스틱 회귀(Logistic Regression)**
  - 모델 결과: [모델 결과 삽입]
  
- **결정 트리(Decision Tree)**
  - 모델 결과: [모델 결과 삽입]
  
- **랜덤 포레스트(Random Forest)**
  - 모델 결과: [모델 결과 삽입]
  
- **그래디언트 부스팅(Gradient Boosting)**
  - 모델 결과: [모델 결과 삽입]
  
- **XGBoost**
  - 모델 결과: [모델 결과 삽입]


각 모델의 결과는 고객 이탈을 예측하는 데 있어 **상당히 높은 정확도**를 기록했으며, 특히 **랜덤 포레스트**와 **XGBoost** 모델은 **ROC-AUC** 점수가 높게 나와, 모델의 **분류 성능**이 우수함을 확인했습니다.

</br>
</br>

**3) 모델 선택 및 결과 분석**

</br>

각 모델의 성능은 교차 검증을 통해 검토되었으며, 주요 평가지표인 **정확도(Accuracy)**, **정밀도(Precision)**, **재현율(Recall)**, **F1 점수** 및 **ROC-AUC**를 기준으로 평가하였습니다.

(결과 캡쳐 사진)

- 정확도(Accuracy) : 예시) 모델들이 대부분 **80% 이상의 정확도**를 보였으며, **고객 이탈 예측 문제**에서 상당히 **신뢰할 수 있는 예측 성능**을 나타냈습니다.</br>

- 정밀도(Precision)와 재현율(Recall) : [결과 설명]

- F1 Score : [결과 설명]

- ROC-AUC : [결과 설명]

</br>


### 6. 팀원 회고
김성근
>
>
대성원
> 
>
윤정연
> 
> 
유수현
>
>
정승연
>
>
</br>
