from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Lasso 모델 (L1 정규화) / Ridge 모델 (L2 정규화)
model = LogisticRegression(penalty='l1', solver='liblinear', C=0.01, max_iter=1000)

# 모델 학습
model.fit(X_train_scaler, y_train)

# 예측
y_preds = model.predict(X_test_scaler)

# 확률 예측 (로지스틱 회귀에서는 predict_proba 사용 가능)
y_probs = model.predict_proba(X_test_scaler)[:, 1] if hasattr(model, "predict_proba") else None

#최적의 하이퍼파라미터 성능 개선 시 / C = 0.01일때 가장 높음을 확인
#param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
#grid_search.fit(X_train_scaler, y_train)

# 결과 출력 (예시)
print(f"Accuracy: {accuracy_score(y_test, y_preds)}")
print(f"Recall: {recall_score(y_test, y_preds, pos_label=1.0)}")
print(f"Precision: {precision_score(y_test, y_preds, pos_label=1.0)}")
print(f"F1 Score: {f1_score(y_test, y_preds, pos_label=1.0)}")
if y_probs is not None:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_probs)}")
print(confusion_matrix(y_test, y_preds))