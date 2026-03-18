import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# 1. Explicitly enable the experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer 

# 2. Now you can import it normally
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import BayesianRidge

# --- 1. 路徑與資料讀取 ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')

train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')

print("正在讀取原始資料...")
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# 分離特徵與目標變數
X_train = train.drop(columns=['id', 'Price'])
y_train = train['Price']
X_test = test.drop(columns=['id'])

# --- 2. 定義特徵分類 ---
numeric_features = ['Compartments', 'Weight Capacity (kg)']
ordinal_features = ['Size', 'Laptop Compartment', 'Waterproof']
size_cats = ['Small', 'Medium', 'Large']
yes_no_cats = ['No', 'Yes']
nominal_features = ['Brand', 'Material', 'Style', 'Color']

# --- 3. 建立專屬處理管線 (Pipelines) ---
num_pipeline = Pipeline(steps=[
    ('imputer', IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=10)),
    ('scaler', RobustScaler())
])

ord_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[size_cats, yes_no_cats, yes_no_cats], 
                               handle_unknown='use_encoded_value', unknown_value=-1))
])

nom_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_features),
        ('ord', ord_pipeline, ordinal_features),
        ('nom', nom_pipeline, nominal_features)
    ],
    remainder='drop'
)

# --- 4. 執行前處理轉換 ---
print("正在執行客製化特徵工程與前處理...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --- 5. 取得轉換後的特徵名稱並轉回 DataFrame ---
nom_feature_names = preprocessor.named_transformers_['nom']['encoder'].get_feature_names_out(nominal_features)
all_feature_names = numeric_features + ordinal_features + list(nom_feature_names)

train_preprocessed = pd.DataFrame(X_train_processed, columns=all_feature_names)
train_preprocessed['Price'] = y_train.values
train_preprocessed['id'] = train['id'].values

test_preprocessed = pd.DataFrame(X_test_processed, columns=all_feature_names)
test_preprocessed['id'] = test['id'].values

# --- 6. 儲存結果 ---
train_preprocessed.to_csv(os.path.join(data_dir, 'train_preprocessed.csv'), index=False)
test_preprocessed.to_csv(os.path.join(data_dir, 'test_preprocessed.csv'), index=False)
print("✅ 前處理完成！已成功儲存為新的 preprocessed.csv，可以交由模型訓練了！")