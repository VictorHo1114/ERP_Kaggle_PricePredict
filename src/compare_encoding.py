import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

# --- 0. 設定圖表風格 ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# --- 1. 路徑與資料讀取 ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
plots_dir = os.path.join(base_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True) 

train_file = os.path.join(data_dir, 'train.csv')
print("正在讀取原始資料...")
train = pd.read_csv(train_file)

X_train = train.drop(columns=['id', 'Price'])
y_train = train['Price']

# --- 2. 定義特徵分類與共通處理 ---
numeric_features = ['Compartments', 'Weight Capacity (kg)']
ordinal_features = ['Size', 'Laptop Compartment', 'Waterproof']
size_cats = ['Small', 'Medium', 'Large']
yes_no_cats = ['No', 'Yes']
nominal_features = ['Brand', 'Material', 'Style', 'Color']

# 數值與序數特徵 (固定使用 IterativeImputer 作為基底)
num_pipeline = Pipeline(steps=[
    ('imputer', IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=10)),
    ('scaler', RobustScaler())
])

ord_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[size_cats, yes_no_cats, yes_no_cats], handle_unknown='use_encoded_value', unknown_value=-1))
])

# ==========================================
# 🛠️ 實驗組與對照組：名目特徵 (Nominal) 的編碼對決
# ==========================================

# 方法 A (對照組): One-Hot Encoding
nom_pipeline_ohe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 方法 B (實驗組): Target Encoding (Scikit-learn 1.3+ 內建)
nom_pipeline_te = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', TargetEncoder(target_type='continuous', random_state=42))
])

preprocessor_ohe = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_features),
        ('ord', ord_pipeline, ordinal_features),
        ('nom', nom_pipeline_ohe, nominal_features)
    ], remainder='drop'
)

preprocessor_te = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_features),
        ('ord', ord_pipeline, ordinal_features),
        ('nom', nom_pipeline_te, nominal_features)
    ], remainder='drop'
)

# ==========================================
# 🏆 執行交叉驗證與評估 (綁定 Model 以防 Data Leakage)
# ==========================================
print("\n=== 比較不同編碼法在 Random Forest 的表現 ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

# 將 Preprocessor 與 Model 綁成完整的 Pipeline
full_pipeline_ohe = Pipeline(steps=[('preprocessor', preprocessor_ohe), ('model', rf)])
full_pipeline_te = Pipeline(steps=[('preprocessor', preprocessor_te), ('model', rf)])

print("正在評估 IterativeImputer + One-Hot Encoding...")
scores_ohe = cross_val_score(full_pipeline_ohe, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
rmse_ohe = -np.mean(scores_ohe)

print("正在評估 IterativeImputer + Target Encoding...")
scores_te = cross_val_score(full_pipeline_te, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
rmse_te = -np.mean(scores_te)

print(f"\nOHE RMSE: {rmse_ohe:.4f}")
print(f"Target Encoding RMSE: {rmse_te:.4f}")

# ==========================================
# 📉 產生報告用圖表
# ==========================================
plt.figure(figsize=(8, 6))
bars = plt.bar(['IterativeImputer +\nOne-Hot Encoding', 'IterativeImputer +\nTarget Encoding'], 
               [rmse_ohe, rmse_te], 
               color=['#5A9BD5', '#ED7D31']) # 換上更有商務感的藍橘配色

# 自動標註數值並微調 Y 軸範圍以凸顯差異
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.002), f'{yval:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

min_val = min(rmse_ohe, rmse_te)
max_val = max(rmse_ohe, rmse_te)
diff = max_val - min_val

# 動態設定 Y 軸範圍，讓圖表差異更具視覺張力
plt.ylim(min_val - (diff * 2), max_val + (diff * 2.5)) 
plt.ylabel('RMSE (Lower is Better)', fontsize=14)
plt.title('Encoding Strategy Impact on Random Forest\n(OHE vs. Target Encoding)', fontsize=16)

save_path = os.path.join(plots_dir, '5_encoding_comparison.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"✅ 編碼比較圖已儲存至 {save_path}")