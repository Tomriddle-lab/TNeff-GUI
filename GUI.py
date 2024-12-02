#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import ttk
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 手动输入数据
# 这里是您可能需要的示例数据，您可以根据实际情况调整
data = {
    'Influent NH4+-N': [10.0],
    'Influent TN': [20.0],
    'Influent Flowrate': [5000.0],
    'DO concentration of Zone 5': [6.0],
    'TN of Zone 5': [10.0],
    'COD of Zone 6': [30.0],
    'TN of Zone 6': [5.0],
    'DO concentration of Zone 7': [3.0],
    'COD of Zone 7': [10.0],
    'TN of Zone 7': [15.0],
    'External Carbon Source': [2.0]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 确保列为数值类型
for feature in df.columns:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

# 执行 scaler.fit
scaler = StandardScaler()
scaler.fit(df)

# 加载预训练模型
model_tn = xgb.Booster()
model_tn.load_model('model_tn.json')

model_power = xgb.Booster()
model_power.load_model('model_power_consumption.json')

# 输入验证函数
def validate_input(value):
    try:
        return float(value)
    except ValueError:
        return None

# 预测函数
def predict():
    inputs = [validate_input(entry.get()) for feature, entry in feature_entry_map.items()]

    if None in inputs:
        output_text.set("Please enter valid numerical values.")
        return
    
    try:
        # 将用户输入转换为带特征名称的 DataFrame
        inputs_df = pd.DataFrame([inputs], columns=df.columns)
        
        # 将用户输入转换为模型可用的格式
        inputs_scaled = scaler.transform(inputs_df)
        dmatrix = xgb.DMatrix(inputs_scaled)
        
        # 进行预测
        effluent_tn_pred = model_tn.predict(dmatrix)[0]
        power_consumption_pred = model_power.predict(dmatrix)[0]

        # 显示预测结果
        optimized_tn.set(f"{effluent_tn_pred:.4f}")
        optimized_power.set(f"{power_consumption_pred:.4f}")

        # 计算节省电能
        original_power_value = float(original_power.get())
        energy_savings = original_power_value - power_consumption_pred
        energy_savings_var.set(f"{energy_savings:.4f}")

    except Exception as e:
        output_text.set(f"An error occurred during prediction:\n{e}")

# 定义单位范围
units = {
    "Influent NH4+-N": "5.4 - 60.7 mg/L",
    "Influent TN": "9.8 - 108.0 mg/L",
    "TN of Zone 5": "4.8 - 44.6 mg/L",
    "TN of Zone 6": "3.0 - 42.3 mg/L",
    "TN of Zone 7": "2.8 - 38.9 mg/L",
    "Influent Flowrate": "0 - 69970.0 m³/d",
    "DO concentration of Zone 5": "5.8 - 8.8 mg/L",
    "COD of Zone 6": "28.0 - 35.5 mg/L",
    "DO concentration of Zone 7": "0.3 - 2.8 mg/L",
    "COD of Zone 7": "0 - 26.4 mg/L",
    "External Carbon Source": "1.9 - 2.3 m³/d"
}

# GUI界面设置
root = tk.Tk()
root.title("Effluent TN & Power Consumption Prediction")
root.geometry("1200x800")
root.configure(bg="#e8f0f2")
default_font = ("Arial", 12)

# 主框架
main_frame = tk.Frame(root, bg="#e8f0f2", padx=20, pady=20)
main_frame.pack(expand=True)

# 标题
title_label = tk.Label(main_frame, text="Effluent TN & Power Consumption Prediction", font=("Arial", 18, "bold"), bg="#e8f0f2")
title_label.grid(row=0, column=0, columnspan=8, pady=15)

# 左列标签和输入框
left_labels = [
    ("Influent NH₄⁺-N", "Influent NH4+-N"),
    ("Influent TN", "Influent TN"),
    ("TN of Zone 5", "TN of Zone 5"),
    ("TN of Zone 6", "TN of Zone 6"),
    ("TN of Zone 7", "TN of Zone 7")
]

# 右列标签和输入框
right_labels = [
    ("Influent Flowrate", "Influent Flowrate"),
    ("DO Concentration of Zone 5", "DO concentration of Zone 5"),
    ("COD of Zone 6", "COD of Zone 6"),
    ("DO Concentration of Zone 7", "DO concentration of Zone 7"),
    ("COD of Zone 7", "COD of Zone 7"),
    ("External Carbon Source", "External Carbon Source")
]

# 创建输入框列表并为标签排序
feature_entry_map = {}

# 左列输入框
for i, (label, feature) in enumerate(left_labels, start=2):
    tk.Label(main_frame, text=label, font=default_font, bg="#e8f0f2").grid(row=i, column=0, padx=10, pady=5, sticky="e")
    entry = tk.Entry(main_frame, font=default_font, width=20)
    entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
    feature_entry_map[feature] = entry
    # 显示单位
    unit_text = units.get(feature, "")
    tk.Label(main_frame, text=f"({unit_text})", font=default_font, bg="#e8f0f2").grid(row=i, column=2, padx=10, pady=5, sticky="w")

# 右列输入框
for i, (label, feature) in enumerate(right_labels, start=2):
    tk.Label(main_frame, text=label, font=default_font, bg="#e8f0f2").grid(row=i, column=3, padx=10, pady=5, sticky="e")
    entry = tk.Entry(main_frame, font=default_font, width=20)
    entry.grid(row=i, column=4, padx=10, pady=5, sticky="ew")
    feature_entry_map[feature] = entry
    # 显示单位
    unit_text = units.get(feature, "")
    tk.Label(main_frame, text=f"({unit_text})", font=default_font, bg="#e8f0f2").grid(row=i, column=5, padx=10, pady=5, sticky="w")

# 添加可控参数标题
tk.Label(main_frame, text="Controllable Parameters", font=("Arial", 14, "bold"), bg="#e8f0f2").grid(row=1, column=3, columnspan=2, pady=10)

# 原始预测结果
original_tn = tk.StringVar()
original_power = tk.StringVar()
optimized_tn = tk.StringVar()
optimized_power = tk.StringVar()
energy_savings_var = tk.StringVar()

# 显示预测按钮
predict_button = tk.Button(main_frame, text="Predict", font=("Arial", 14), command=predict, bg="#4CAF50", fg="white", width=18)
predict_button.grid(row=8, column=0, columnspan=8, pady=20)

# 结果显示框
output_frame = tk.Frame(main_frame, bg="#e8f0f2")
output_frame.grid(row=9, column=0, columnspan=8, padx=10, pady=20)

tk.Label(output_frame, text="Original Effluent TN (mg/L):", font=default_font, bg="#e8f0f2").grid(row=0, column=0, sticky="e")
tk.Entry(output_frame, textvariable=original_tn, font=default_font, state="readonly").grid(row=0, column=1)

tk.Label(output_frame, text="Optimized Effluent TN (mg/L):", font=default_font, bg="#e8f0f2").grid(row=0, column=2, sticky="e")
tk.Entry(output_frame, textvariable=optimized_tn, font=default_font, state="readonly").grid(row=0, column=3)

tk.Label(output_frame, text="Original Power Consumption (kWh):", font=default_font, bg="#e8f0f2").grid(row=1, column=0, sticky="e")
tk.Entry(output_frame, textvariable=original_power, font=default_font, state="readonly").grid(row=1, column=1)

tk.Label(output_frame, text="Optimized Power Consumption (kWh):", font=default_font, bg="#e8f0f2").grid(row=1, column=2, sticky="e")
tk.Entry(output_frame, textvariable=optimized_power, font=default_font, state="readonly").grid(row=1, column=3)

tk.Label(output_frame, text="Energy Savings (kWh):", font=default_font, bg="#e8f0f2").grid(row=2, column=0, sticky="e")
tk.Entry(output_frame, textvariable=energy_savings_var, font=default_font, state="readonly").grid(row=2, column=1)

root.mainloop()


# In[ ]:




