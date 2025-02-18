#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing  import StandardScaler
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot  as plt
import shap

# Load dataset
file_path = r'C:\path\to\your\data.csv'  
df = pd.read_csv(file_path,  encoding='GBK')

# Split features and target
X = df.drop(columns='Effluent  TN')
y = df['Effluent TN']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost model
xgboost_model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Model training
xgboost_model.fit(X_train,  y_train)

# Plot feature importance
feature_importances = xgboost_model.feature_importances_ 
sorted_idx = np.argsort(feature_importances)[::-1] 

plt.figure(figsize=(12,  8))
plt.bar(range(len(feature_importances)),  feature_importances[sorted_idx], align='center', color='deepskyblue', alpha=0.3)
plt.xticks(range(len(feature_importances)),  [X.columns[i] for i in sorted_idx], rotation=90, fontsize=18)
plt.title('Feature  Importance', fontsize=20)
plt.xlabel('Features',  fontsize=20)
plt.ylabel('Importance',  fontsize=20)
plt.show() 

# SHAP value analysis
explainer = shap.TreeExplainer(xgboost_model)
shap_values = explainer.shap_values(X_train) 
shap.summary_plot(shap_values,  X_train, feature_names=X.columns) 

