#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.linear_model  import LinearRegression 
from sklearn.model_selection  import KFold, train_test_split 
from sklearn.metrics  import mean_squared_error, r2_score 
import numpy as np 
import pandas as pd
import matplotlib.pyplot  as plt 
import statsmodels.api  as sm 
from sklearn.preprocessing  import StandardScaler 
import pickle 
 
# Read data from CSV file 
file_path = r'C:\path\to\your\data.csv'
df = pd.read_csv(file_path,  encoding='GBK')
 
# Extract features and target variable 
X = df.drop(columns='Effluent  TN')
y = df['Effluent TN']
 
# Split data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
 
# Standardize features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
 
# Save the scaler for future use 
with open('scaler.pkl',  'wb') as f:
    pickle.dump(scaler,  f)
# Initialize model 
mlr_model = LinearRegression()
 
# Define KFold 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
 
# Store evaluation results 
train_r2_scores = []
train_rmse_scores = []
test_r2_scores = []
test_rmse_scores = []
 
# Store data for visualization 
all_y_train_pred = []
all_y_test_pred = []
all_y_train_true = []
all_y_test_true = []
 
# Cross-validation loop 
for train_index, test_index in kf.split(X_train_scaled): 
    X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index],  y_train.iloc[test_index] 
    
    # Train model 
    mlr_model.fit(X_train_fold,  y_train_fold)
    
    # Predict 
    y_train_pred = mlr_model.predict(X_train_fold) 
    y_test_pred = mlr_model.predict(X_test_fold) 
    
    # Collect results 
    all_y_train_pred.extend(y_train_pred) 
    all_y_test_pred.extend(y_test_pred) 
    all_y_train_true.extend(y_train_fold) 
    all_y_test_true.extend(y_test_fold) 
    
    # Calculate metrics 
    train_r2 = r2_score(y_train_fold, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_fold,  y_train_pred))
    test_r2 = r2_score(y_test_fold, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_fold,  y_test_pred))
    
    train_r2_scores.append(train_r2) 
    train_rmse_scores.append(train_rmse) 
    test_r2_scores.append(test_r2) 
    test_rmse_scores.append(test_rmse) 
 
# Calculate averages 
avg_train_r2 = np.mean(train_r2_scores) 
avg_train_rmse = np.mean(train_rmse_scores) 
avg_test_r2 = np.mean(test_r2_scores) 
avg_test_rmse = np.mean(test_rmse_scores) 
 
print("\nAverage Training Set Results:")
print("Average R²:", avg_train_r2)
print("Average RMSE:", avg_train_rmse)
 
print("\nAverage Test Set Results:")
print("Average R²:", avg_test_r2)
print("Average RMSE:", avg_test_rmse)
 
# Visualization 
plt.figure(figsize=(10,  6), dpi=600)
plt.scatter(all_y_train_true,  all_y_train_pred,
            edgecolors='black', c='darkgreen', marker='^', s=100, alpha=0.6, linewidth=1, label='Training Data')
plt.scatter(all_y_test_true,  all_y_test_pred,
            edgecolors='black', c='lightyellow', marker='o', s=100, alpha=0.6, linewidth=1, label='Test Data')
 
# Fit and plot regression line 
X_test = np.array(all_y_test_true).reshape(-1,  1)
y_test_pred_selected = np.array(all_y_test_pred) 
 
sorted_indices = np.argsort(X_test.flatten()) 
X_test_sorted = X_test[sorted_indices]
y_test_pred_sorted = y_test_pred_selected[sorted_indices]
 
linear_model = sm.OLS(y_test_pred_sorted, sm.add_constant(X_test_sorted)).fit() 
y_pred_sorted = linear_model.predict(sm.add_constant(X_test_sorted)) 
 
# Get and plot confidence interval 
predictions_summary = linear_model.get_prediction(sm.add_constant(X_test_sorted)).summary_frame(alpha=0.05) 
 
plt.plot(X_test_sorted,  y_pred_sorted, color='red', linewidth=2, label='Fitted Line')
plt.fill_between(X_test_sorted.flatten(), 
                predictions_summary['obs_ci_lower'],
                predictions_summary['obs_ci_upper'],
                color='lightpink', alpha=0.3, label='95% CI')
 
# Add ideal line 
plt.plot([0,  17.5], [0, 17.5], c='blue', linestyle='--', lw=2, label='Ideal Line')
 
plt.xlim([0,  17.5])
plt.ylim([0,  17.5])
plt.legend(fontsize=12) 
plt.tick_params(axis='both',  which='major', labelsize=10)
plt.show() 

