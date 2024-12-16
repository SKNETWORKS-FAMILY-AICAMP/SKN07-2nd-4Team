from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

params = {
        "n_estimators": [10, 50, 100, 200, 300],  # 모델 생성 트리 개수
        "learning_rate": [0.1],                   # 학습률
        "max_depth": [1, 2, 3, 4, 5],             # 트리 최대 깊이
        "subsample": [0.5, 0.7],                  # sample rate
}

model = GradientBoostingClassifier()
model.fit(X_train_scaler, y_train)

#GridSearchCV 사용 (교차 검증을 통한 하이퍼파라미터 최적화)
grid_search = GridSearchCV(
    estimator=model,          
    param_grid=params,  
    scoring='accuracy',
    refit='accuracy',
    cv=5,             
    n_jobs=-1,         
)

grid_search.fit(X_train_scaler, y_train)

best_param_ = grid_search.best_params_
best_model = grid_search.best_estimator_

# 최적의 하이퍼파라미터 출력
print("Best Parameters:", best_param_)
print("Best models:", best_model)

y_preds = best_model.predict(X_test_scaler)
y_probs = best_model.predict_proba(X_test_scaler)[:, 1] if hasattr(best_model, "predict_proba") else None

# 결과 출력 (예시)
print(f"Accuracy: {accuracy_score(y_test, y_preds)}")
print(f"Recall: {recall_score(y_test, y_preds, pos_label=1.0)}")
print(f"Precision: {precision_score(y_test, y_preds, pos_label=1.0)}")
print(f"F1 Score: {f1_score(y_test, y_preds, pos_label=1.0)}")
if y_probs is not None:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_probs)}")
print(confusion_matrix(y_test, y_preds))