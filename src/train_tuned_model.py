import pandas as pd
import numpy as np
import os
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import root_mean_squared_error, r2_score

# --- 1. 路徑與資料讀取 ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
submissions_dir = os.path.join(base_dir, 'submissions')

train_file = os.path.join(data_dir, 'train_preprocessed.csv')
test_file = os.path.join(data_dir, 'test_preprocessed.csv')

print("Loading preprocessed data...")
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

X_train = train.drop(columns=['id', 'Price'])
y_train = train['Price']
X_test = test.drop(columns=['id'])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- 2. Ridge Regression 訓練 ---
print("\n=== Tuning Ridge Regression (L2 Regularization) ===")
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
ridge = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
ridge.fit(X_train, y_train)

print(f"Best Alpha (regularization strength): {ridge.alpha_}")
y_train_pred_ridge = ridge.predict(X_train)
r2_ridge = r2_score(y_train, y_train_pred_ridge)
print(f"Ridge R^2 on training set: {r2_ridge:.4f}")

cv_scores_ridge = cross_val_score(ridge, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
print(f"Ridge 5-Fold CV Average RMSE: {-np.mean(cv_scores_ridge):.4f}")

# --- 3. Random Forest Regressor 訓練 ---
print("\n=== Training Random Forest Regressor (Non-linear) ===")
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_train_pred_rf = rf.predict(X_train)
r2_rf = r2_score(y_train, y_train_pred_rf)
print(f"Random Forest R^2 on training set: {r2_rf:.4f}")

cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
print(f"Random Forest 5-Fold CV Average RMSE: {-np.mean(cv_scores_rf):.4f}")

# --- 4. 預測與輸出 Kaggle Submission ---
print("\nGenerating final predictions using Random Forest...")
y_test_pred_rf = rf.predict(X_test)

submission_file = os.path.join(submissions_dir, 'submission_tuned.csv')
submission = pd.DataFrame({
    'id': test['id'],
    'Price': y_test_pred_rf
})
submission.to_csv(submission_file, index=False)
print(f"✅ Saved tuned predictions to {submission_file}")