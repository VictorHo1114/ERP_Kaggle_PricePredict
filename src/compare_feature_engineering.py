import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, TargetEncoder
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
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

# --- 2. 定義特徵分類與共通處理 (目前最強的 Baseline) ---
numeric_features = ['Compartments', 'Weight Capacity (kg)']
ordinal_features = ['Size', 'Laptop Compartment', 'Waterproof']
size_cats = ['Small', 'Medium', 'Large']
yes_no_cats = ['No', 'Yes']
nominal_features = ['Brand', 'Material', 'Style', 'Color']

# 數值特徵：IterativeImputer + RobustScaler
num_pipeline = Pipeline(steps=[
    ('imputer', IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=10)),
    ('scaler', RobustScaler())
])

# 序數特徵：眾數補值 + OrdinalEncoder
ord_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[size_cats, yes_no_cats, yes_no_cats], handle_unknown='use_encoded_value', unknown_value=-1))
])

# 名目特徵：常數補值 + TargetEncoder (使用你實驗出的最佳解)
nom_pipeline_te = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', TargetEncoder(target_type='continuous', random_state=42))
])

# 核心前處理器：我們所有實驗的基準
base_preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_features),
        ('ord', ord_pipeline, ordinal_features),
        ('nom', nom_pipeline_te, nominal_features)
    ], remainder='drop'
)

# ==========================================
# 🛠️ 實驗組與對照組：看看下游模型會不會進步
# ==========================================
# 下游共用模型
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

# 方法 A (對照組): 單純的 Baseline (TE + IterativeImputer)
pipeline_baseline = Pipeline(steps=[
    ('preprocessor', base_preprocessor),
    ('model', rf_model)
])

# 方法 B (實驗組 1): 特徵選擇 (Feature Selection)
# 概念：用一棵小的隨機森林先算特徵重要性，把不重要的雜訊砍掉，再餵給下游模型
pipeline_selection = Pipeline(steps=[
    ('preprocessor', base_preprocessor),
    ('selector', SelectFromModel(RandomForestRegressor(n_estimators=30, random_state=42), threshold='median')),
    ('model', rf_model)
])

# 方法 C (實驗組 2): 特徵提取 (Feature Extraction - PCA)
# 概念：把特徵壓縮成幾個主成分，消除共線性
pipeline_pca = Pipeline(steps=[
    ('preprocessor', base_preprocessor),
    ('pca', PCA(n_components=0.95, random_state=42)), # 保留 95% 的變異資訊
    ('model', rf_model)
])

# ==========================================
# 🏆 執行交叉驗證與評估 (CV=5)
# ==========================================
print("\n=== 比較加入特徵選擇/提取對下游模型 (Random Forest) 的影響 ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_pipeline(name, pipeline):
    print(f"正在評估 {name}...")
    scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
    rmse = -np.mean(scores)
    return rmse

rmse_baseline = evaluate_pipeline("Baseline (目前最強組合)", pipeline_baseline)
rmse_selection = evaluate_pipeline("Baseline + 特徵選擇 (SelectFromModel)", pipeline_selection)
rmse_pca = evaluate_pipeline("Baseline + 特徵提取 (PCA)", pipeline_pca)

print(f"\n📊 最終結果:")
print(f"1. Baseline RMSE:       {rmse_baseline:.4f}")
print(f"2. + Feature Selection: {rmse_selection:.4f}")
print(f"3. + PCA Extraction:    {rmse_pca:.4f}")

# ==========================================
# 📉 產生報告用圖表
# ==========================================
plt.figure(figsize=(10, 7))
labels = ['Baseline\n(TE + Imputer)', 'Feature Selection\n(Tree Based)', 'Feature Extraction\n(PCA)']
rmse_values = [rmse_baseline, rmse_selection, rmse_pca]
colors = ['#8C92AC', '#5A9BD5', '#ED7D31'] # 灰(基準)、藍(選擇)、橘(提取)

bars = plt.bar(labels, rmse_values, color=colors, width=0.6)

# 自動標註數值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.002), f'{yval:.4f}', 
             ha='center', va='bottom', fontsize=13, fontweight='bold')

min_val = min(rmse_values)
max_val = max(rmse_values)
diff = max_val - min_val

# 動態設定 Y 軸範圍，讓圖表差異更具視覺張力
plt.ylim(min_val - (diff * 1.5), max_val + (diff * 2.5)) 
plt.ylabel('RMSE (Lower is Better)', fontsize=14)
plt.title('Impact of Feature Engineering on Downstream Model\n(Baseline vs. Selection vs. PCA)', fontsize=16)

save_path = os.path.join(plots_dir, '6_feature_engineering_comparison.png')
plt.savefig(save_path, bbox_inches='tight')
plt.close()
print(f"\n✅ 特徵工程比較圖已儲存至 {save_path}")