# -Badminton-Player-Performance-Dataset
<img width="251" height="800" alt="image" src="https://github.com/user-attachments/assets/8e35f92e-75fa-4239-b734-59b54b724a71" />

 <img width="400" height="208" alt="image" src="https://github.com/user-attachments/assets/135518a7-12eb-42a7-8a1b-a2fdd38b0b15" />

 <img width="448" height="387" alt="image" src="https://github.com/user-attachments/assets/346ff3c3-2b1c-449c-b5fc-4c17414d6760" />

🏸 Machine Learning Model Architecture & Performance Overview
This Machine Learning pipeline predicts a badminton player's overall BWF Ranking Points ( 
y
^
​	
 ) based on physical attributes, demographic markers, and match engagement stats.
🏗️ Model Blueprint & Pipeline Mechanics
🎯 Target Variable (Points): Predicts total BWF ranking points accrued across competitive tournament circuits.
🛡️ Leakage Shielding: Explicitly drops Rank (which has a near-perfect negative correlation of r=−0.93 with points) to prevent target leakage, alongside Name (an unstructured row identifier).
⚙️ Numerical Feature Transformation: Features like Tournaments, Matches_Played, Age, Height_cm, and Weight_kg undergo StandardScaler normalization to standardize variance scales (μ=0,σ=1).
🌐 Categorical Encoding: Attributes Nationality (36 high-cardinality countries) and Handedness (Left vs. Right) are processed via OneHotEncoder(handle_unknown='ignore') into robust binary indicator vectors.
🌲 Core Estimator: Uses a Random Forest Regressor ensemble with optimized hyperparameter constraints (max_depth, n_estimators, min_samples_split) tuned via 5-fold cross-validation.
📊 Performance & Importance Breakdown
Metric 📐	Score / Value 📌	Interpretation 💡
Primary Metric (R 
2
 )	0.12	Captures baseline variance without relying on explicit rank signals.
Mean Absolute Error (MAE)	~14,641 Points	Average magnitude of point prediction deviations.
Top Feature Driver 🏆	Tournaments Played (~25.8%)	Volume of tournament participation is the single strongest driver of ranking points.
Secondary Feature Driver ⚔️	Matches Played (~19.3%)	Match density reflects how far players progress in tournament draws.
Demographic Signal 🌍	Nationality (~15.9%)	Strong national badminton programs (e.g., China, Denmark, Japan) carry significant predictive weight.
💻 Complete Google Colab Code Setup
Run each block below in separate code cells inside your Google Colab environment.
📍 Cell 1: Environment Setup & Dataset Ingestion 📥
Python
# -------------------------------------------------------------
# 📥 CELL 1: Import Libraries & Load Data
# -------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📂 Load the dataset
file_name = "1. Dataset-selected-columns.csv"
df = pd.read_csv(file_name)

print("✅ Dataset successfully loaded!")
print(f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
display(df.head(5))
📍 Cell 2: Exploratory Data Analysis & Leakage Auditing 🔎
Python
# -------------------------------------------------------------
# 🔎 CELL 2: Exploratory Data Analysis (EDA) & Correlations
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("🔥 Feature Correlation Matrix", fontsize=14, fontweight='bold')
plt.show()

# 🎯 Distribution of Target Variable (Points)
plt.figure(figsize=(8, 4))
sns.histplot(df['Points'], bins=20, kde=True, color='teal')
plt.title("📈 Distribution of BWF Ranking Points", fontsize=12)
plt.xlabel("Points 🏆")
plt.ylabel("Frequency 📊")
plt.show()
📍 Cell 3: Feature Engineering & Preprocessing Pipeline ⚙️
Python
# -------------------------------------------------------------
# ⚙️ CELL 3: Data Preprocessing & Pipeline Construction
# -------------------------------------------------------------
# 🛡️ Drop Rank (Leakage) and Name (Identifier)
X = df.drop(columns=['Points', 'Rank', 'Name'])
y = df['Points']

# 🏷️ Group column types
categorical_cols = ['Nationality', 'Handedness']
numerical_cols = ['Tournaments', 'Age', 'Height_cm', 'Weight_kg', 'Matches_Played']

# 🔀 Construct Preprocessing Pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# ✂️ Train-Test Split (80% Train / 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"🎯 Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
📍 Cell 4: Model Benchmarking Suite 🏎️
Python
# -------------------------------------------------------------
# 🏎️ CELL 4: Evaluate Baseline Machine Learning Models
# -------------------------------------------------------------
candidate_models = {
    "🎯 Ridge Regression": Ridge(alpha=10.0),
    "🌲 Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "🚀 Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

benchmark_results = {}

for model_name, model_obj in candidate_models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_obj)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    benchmark_results[model_name] = {
        'MAE 📉': round(mean_absolute_error(y_test, preds), 2),
        'RMSE 📊': round(np.sqrt(mean_squared_error(y_test, preds)), 2),
        'R² Score 🎯': round(r2_score(y_test, preds), 4)
    }

results_table = pd.DataFrame(benchmark_results).T
display(results_table.sort_values(by="R² Score 🎯", ascending=False))
📍 Cell 5: Hyperparameter Tuning & Optimal Model Extraction 🎛️
Python
# -------------------------------------------------------------
# 🎛️ CELL 5: Hyperparameter Optimization via GridSearchCV
# -------------------------------------------------------------
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 🛠️ Define tuning grid
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 5, 10],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

# 🔍 Run Grid Search CV
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("🎉 Hyperparameter Tuning Complete!")
print("🏆 Best Configuration:", grid_search.best_params_)
📍 Cell 6: Feature Importance Analysis & Final Diagnostics 🚀
Python
# -------------------------------------------------------------
# 🚀 CELL 6: Model Evaluation & Visualizing Feature Importances
# -------------------------------------------------------------
y_pred_best = best_model.predict(X_test)

r2_final = r2_score(y_test, y_pred_best)
mae_final = mean_absolute_error(y_test, y_pred_best)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_best))

print(f"📌 Final Optimized R² Score: {r2_final:.4f}")
print(f"📉 Final Mean Absolute Error (MAE): {mae_final:.2f} Points")
print(f"📊 Final Root Mean Squared Error (RMSE): {rmse_final:.2f} Points\n")

# 🌟 Feature Importances Extraction
cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat']
encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_cols))
all_feature_names = numerical_cols + encoded_cat_features

importances = best_model.named_steps['regressor'].feature_importances_
feature_df = pd.DataFrame({'Feature 🏷️': all_feature_names, 'Importance ⚡': importances})
feature_df = feature_df.sort_values(by='Importance ⚡', ascending=False)

# 🎨 Plot Top 10 Features
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance ⚡', y='Feature 🏷️', data=feature_df.head(10), palette='magma')
plt.title("🌟 Top 10 Most Influential Performance Features", fontsize=13, fontweight='bold')
plt.xlabel("Relative Feature Importance Weight")
plt.show()
