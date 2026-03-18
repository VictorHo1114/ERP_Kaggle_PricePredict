import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import root_mean_squared_error

# --- 0. 設定圖表風格 ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# --- 1. 路徑與資料讀取 ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
plots_dir = os.path.join(base_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True) # 建立資料夾存圖

train_file = os.path.join(data_dir, 'train.csv')
print("正在讀取原始資料...")
train = pd.read_csv(train_file)

X_train = train.drop(columns=['id', 'Price'])
y_train = train['Price']

# ==========================================
# 📊 第一階段：原始資料缺失狀況視覺化
# ==========================================
print("\n=== 產生缺失值視覺化圖表 ===")
# 1. 缺失值長條圖 (看數量)
fig_bar = plt.figure(figsize=(10, 6))
msno.bar(X_train, color="dodgerblue", figsize=(10, 6), fontsize=12)
plt.title("Missing Values Count per Feature", fontsize=16)
plt.savefig(os.path.join(plots_dir, '1_missing_bar.png'), bbox_inches='tight')
plt.close()

# 2. 缺失值矩陣圖 (看分佈與關聯)
fig_matrix = plt.figure(figsize=(10, 6))
msno.matrix(X_train, color=(0.1, 0.2, 0.5), figsize=(10, 6), fontsize=12)
plt.title("Missing Values Matrix (Pattern Identification)", fontsize=16)
plt.savefig(os.path.join(plots_dir, '2_missing_matrix.png'), bbox_inches='tight')
plt.close()
print("✅ 缺失值圖表已儲存至 plots 資料夾")


# ==========================================
# 🛠️ 第二階段：建立兩種不同的 Pipeline
# ==========================================
numeric_features = ['Compartments', 'Weight Capacity (kg)']
ordinal_features = ['Size', 'Laptop Compartment', 'Waterproof']
size_cats = ['Small', 'Medium', 'Large']
yes_no_cats = ['No', 'Yes']
nominal_features = ['Brand', 'Material', 'Style', 'Color']

# 共通的類別特徵處理
ord_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[size_cats, yes_no_cats, yes_no_cats], handle_unknown='use_encoded_value', unknown_value=-1))
])
nom_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 差異點：數值特徵的 Imputer
# 方法 A: SimpleImputer (平均數)
num_pipeline_simple = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])

# 方法 B: IterativeImputer (多變數迴歸)
num_pipeline_iterative = Pipeline(steps=[
    ('imputer', IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=10)),
    ('scaler', RobustScaler())
])

preprocessor_simple = ColumnTransformer(
    transformers=[
        ('num', num_pipeline_simple, numeric_features),
        ('ord', ord_pipeline, ordinal_features),
        ('nom', nom_pipeline, nominal_features)
    ], remainder='drop'
)

preprocessor_iterative = ColumnTransformer(
    transformers=[
        ('num', num_pipeline_iterative, numeric_features),
        ('ord', ord_pipeline, ordinal_features),
        ('nom', nom_pipeline, nominal_features)
    ], remainder='drop'
)

# 執行轉換
X_simple = preprocessor_simple.fit_transform(X_train)
X_iterative = preprocessor_iterative.fit_transform(X_train)

# ==========================================
# 📉 第三階段：資料分佈對比視覺化 (KDE Plot)
# ==========================================
print("\n=== 產生填補後資料分佈對比圖 ===")
# 為了畫圖，我們單獨把 Weight Capacity 取出來比較 (假設它有缺失值)
target_feature = 'Weight Capacity (kg)'
if X_train[target_feature].isnull().sum() > 0:
    plt.figure(figsize=(10, 6))
    
    # 原始資料 (排除 NaN)
    sns.kdeplot(X_train[target_feature].dropna(), label='Original Data (Drop NaN)', color='black', linewidth=2, linestyle='--')
    
    # SimpleImputer 填補後 (需要反向縮放以便和原始資料在同一量級比較，此處簡化直接取出 Imputer 的結果)
    simple_imputed_data = SimpleImputer(strategy='mean').fit_transform(X_train[[target_feature]])
    sns.kdeplot(simple_imputed_data.flatten(), label='SimpleImputer (Mean)', color='red', alpha=0.7)
    
    # IterativeImputer 填補後 (利用所有數值欄位填補)
    iter_imputed_data = IterativeImputer(estimator=BayesianRidge(), random_state=42).fit_transform(X_train[numeric_features])
    target_idx = numeric_features.index(target_feature)
    sns.kdeplot(iter_imputed_data[:, target_idx], label='IterativeImputer (BayesianRidge)', color='green', alpha=0.7)
    
    plt.title(f"Distribution Comparison: {target_feature}", fontsize=16)
    plt.xlabel(target_feature)
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, '3_distribution_comparison.png'), bbox_inches='tight')
    plt.close()
    print("✅ 資料分佈對比圖已儲存")

# ==========================================
# 🏆 第四階段：下游任務 (模型訓練) 表現比較
# ==========================================
print("\n=== 比較下游模型表現 (Random Forest Cross Validation) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

# 計算 SimpleImputer 的 RMSE
scores_simple = cross_val_score(rf, X_simple, y_train, cv=kf, scoring='neg_root_mean_squared_error')
rmse_simple = -np.mean(scores_simple)

# 計算 IterativeImputer 的 RMSE
scores_iterative = cross_val_score(rf, X_iterative, y_train, cv=kf, scoring='neg_root_mean_squared_error')
rmse_iterative = -np.mean(scores_iterative)

print(f"SimpleImputer RMSE: {rmse_simple:.4f}")
print(f"IterativeImputer RMSE: {rmse_iterative:.4f}")

# 畫出對比長條圖
plt.figure(figsize=(8, 6))
bars = plt.bar(['SimpleImputer\n(Mean)', 'IterativeImputer\n(Multivariate)'], 
               [rmse_simple, rmse_iterative], 
               color=['lightcoral', 'mediumseagreen'])

# 在柱狀圖上方加上數值標籤
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.005), f'{yval:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylim(min(rmse_simple, rmse_iterative) * 0.95, max(rmse_simple, rmse_iterative) * 1.05) # 微調 Y 軸讓差異更明顯
plt.ylabel('RMSE (Lower is Better)', fontsize=14)
plt.title('Downstream Model Performance Comparison\n(Random Forest RMSE)', fontsize=16)
plt.savefig(os.path.join(plots_dir, '4_model_performance_comparison.png'), bbox_inches='tight')
plt.close()
print("✅ 模型表現比較圖已儲存至 plots 資料夾")
print("\n🎉 程式執行完畢！趕快去 plots 資料夾查看你的精美圖表吧！")