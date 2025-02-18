#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot  as plt
from xgboost import XGBRegressor
from sklearn.model_selection  import train_test_split
from sklearn.inspection  import PartialDependenceDisplay

# Read data from CSV file
file_path = r'C:\path\to\your\data.csv'  
df = pd.read_csv(file_path,  encoding='GBK')

# Split features and target
X = df.drop(columns='Effluent  TN')
y = df['Effluent TN']

# Split dataset into training and testing sets
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

# Train the model
xgboost_model.fit(X_train,  y_train)

# Feature selection and plot 2D PDP，Complementing other target features and related features
target_feature = 'External Carbon Source'
related_features = ['COD of Zone 6', 'COD of Zone 7']

for feature in related_features:
    if feature == target_feature:
        continue
    feature_idx = X.columns.get_loc(feature) 
    target_idx = X.columns.get_loc(target_feature) 
    
    plt.figure(figsize=(8,  6))
    PartialDependenceDisplay.from_estimator( 
        xgboost_model, X_train, [(target_idx, feature_idx)], 
        feature_names=X.columns,  kind='average'
    )
    plt.title(f'2D  Partial Dependence: {target}_feature vs {feature}')
    plt.savefig(f'./2d_pdp_{target_feature}_{feature}.png',  dpi=300, bbox_inches='tight')
    plt.show() 

