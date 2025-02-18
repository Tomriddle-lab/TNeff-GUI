#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot  as plt
from xgboost import XGBRegressor
from sklearn.model_selection  import train_test_split
from sklearn.inspection  import partial_dependence
from scipy.interpolate  import splev, splrep

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

# List of features to plot
features_to_plot = [
    'External Carbon Source',
    'COD of Zone 6',
    'COD of Zone 7',
    'DO of Zone 7',
    'DO of Zone 5',
    'Influent NH4+-N',
    'Influent TN',
    'Influent Flowrate',
    'TN of Zone 5',
    'TN of Zone 6',
    'TN of Zone 7',
    'Energy Consumption'
]

# Set Seaborn theme
sns.set_theme(style="ticks",  palette="deep", font_scale=1.1)

def plot_pdp(feature):
    # Calculate Partial Dependence and Individual Conditional Expectation
    pdp = partial_dependence(xgboost_model, X_train, [feature], kind="both", grid_resolution=50)
    
    # Prepare data for plotting
    plot_x = pd.Series(pdp.grid_values[0]).rename('x') 
    plot_y = pd.Series(pdp.average[0]).rename('y') 
    plot_i = pdp.individual[0] 

    # Smooth interpolation
    tck = splrep(plot_x, plot_y, s=30)
    xnew = np.linspace(plot_x.min(),  plot_x.max(),  300)
    ynew = splev(xnew, tck, der=0)

    # Create plot
    fig, ax = plt.subplots(figsize=(8,  6))
    
    # Plot ICE curves
    for a in plot_i:
        a_series = pd.Series(a)
        df_i = pd.concat([plot_x,  a_series.rename('y')],  axis=1)
        sns.lineplot(data=df_i,  x="x", y="y", color='k', linewidth=1.5, linestyle='--', alpha=0.6, ax=ax)
    
    # Plot smoothed PDP
    ax.plot(xnew,  ynew, color='peru', linewidth=2, label='Smoothed PDP')
    
    # Add confidence interval
    std_error = np.std(plot_y)  / np.sqrt(len(plot_y)) 
    lower_bound = plot_y - 1.96 * std_error
    upper_bound = plot_y + 1.96 * std_error
    ax.fill_between(plot_x,  lower_bound, upper_bound, color='khaki', alpha=0.3, label='95% CI')
    
    # Add rug plot
    sns.rugplot(data=X_train.sample(100),  x=feature, height=0.05, color='k', alpha=0.3, ax=ax)
    
    # Set labels and limits
    ax.set_ylabel('Partial  Dependence')
    ax.set_xlabel(feature) 
    
    x_min = plot_x.min()  - 0.1*(plot_x.max()  - plot_x.min()) 
    x_max = plot_x.max()  + 0.1*(plot_x.max()  - plot_x.min()) 
    ax.set_xlim(x_min,  x_max)
    
    # Add legend
    ax.legend() 
    
    # Save and show plot
    plt.savefig(f'./pdpplot_{feature}.png',  dpi=900, bbox_inches='tight')
    plt.show() 

# Plot PDP for each feature
for feature in features_to_plot:
    plot_pdp(feature)

