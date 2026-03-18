import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
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

# Separate features and target
X_train = train.drop(columns=['id', 'Price'])
y_train = train['Price']
X_test = test.drop(columns=['id'])

# --- 2. 模型訓練與驗證 ---
print("Initializing Linear Regression Model...")
model = LinearRegression()

print("Performing 5-Fold Cross Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
avg_rmse = np.mean(-cv_scores)
print(f"5-Fold Cross-Validation Average RMSE: {avg_rmse:.4f}")

print("\nTraining model on the entire dataset...")
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
r2 = r2_score(y_train, y_train_pred)
print(f"R-squared (R^2) score on training set: {r2:.4f}")

# --- 3. 預測與輸出 Kaggle Submission ---
print("\nGenerating predictions for the test set...")
y_test_pred = model.predict(X_test)

submission_file = os.path.join(submissions_dir, 'submission_linear.csv')
print(f"Saving predictions to {submission_file}...")
submission = pd.DataFrame({
    'id': test['id'],
    'Price': y_test_pred
})
submission.to_csv(submission_file, index=False)

print("✅ Model training and prediction script completed successfully!")