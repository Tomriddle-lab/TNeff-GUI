#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import ttk
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Default data
default_data = {
    'Influent NH4+-N': 29.6,
    'Influent TN': 35.9,
    'Influent Flowrate': 38808.4,
    'DO of Zone 5': 7.1,
    'TN of Zone 5': 22.2,
    'COD of Zone 6': 55.0,
    'TN of Zone 6': 13.8,
    'DO of Zone 7': 4.06,
    'COD of Zone 7': 31.9,
    'TN of Zone 7': 11.7,
    'External Carbon Source': 3.3
}

df = pd.DataFrame([default_data])

scaler = StandardScaler()
scaler.fit(df)

model_tn = xgb.Booster()
model_tn.load_model('model_effluent_TN.json')

model_power = xgb.Booster()
model_power.load_model('model_energy_consumption.json')

# Input validation function
def validate_input(value):
    try:
        return float(value)
    except ValueError:
        return None

# Variable to limit the number of runs
run_count = 0
first_external_carbon_source = None

# Reset input fields
def clear_inputs():
    global run_count, first_external_carbon_source
    run_count = 0
    first_external_carbon_source = None
    for entry in feature_entry_map.values():
        entry.delete(0, tk.END)
    original_tn.set("")
    original_power.set("")
    optimized_tn.set("")
    optimized_power.set("")
    energy_savings_var.set("")
    carbon_savings_var.set("")
    output_text.set("")

# Prediction function
def predict():
    global run_count, first_external_carbon_source

    if run_count >= 2:
        output_text.set("The prediction limit has been reached. Please click the Reset button to reset and try again.")
        return

    # Collect user inputs, use default values if empty
    inputs = []
    for feature, entry in feature_entry_map.items():
        value = entry.get().strip()  # Get user input
        if not value:  # If empty
            inputs.append(default_data[feature])  # Use default value
        else:
            validated = validate_input(value)
            if validated is None:  # Show error if validation fails
                output_text.set(f"Invalid input for {feature}. Please enter a valid number.")
                return
            inputs.append(validated)

    try:
        # Convert input data to DataFrame
        inputs_df = pd.DataFrame([inputs], columns=default_data.keys())
        inputs_scaled = scaler.transform(inputs_df)
        dmatrix = xgb.DMatrix(inputs_scaled)

        # Prediction results
        effluent_tn_pred = model_tn.predict(dmatrix)[0]
        power_consumption_pred = model_power.predict(dmatrix)[0]

        # First prediction results
        if run_count == 0:
            original_tn.set(f"{effluent_tn_pred:.4f}")
            original_power.set(f"{power_consumption_pred:.4f}")
            first_external_carbon_source = inputs_df["External Carbon Source"].iloc[0]
        # Second prediction results
        elif run_count == 1:
            optimized_tn.set(f"{effluent_tn_pred:.4f}")
            optimized_power.set(f"{power_consumption_pred:.4f}")

            # Calculate energy savings
            original_power_value = float(original_power.get()) if original_power.get() else 0.0
            energy_savings = original_power_value - power_consumption_pred
            energy_savings_var.set(f"{energy_savings:.4f}")

            # Calculate carbon source savings
            second_external_carbon_source = inputs_df["External Carbon Source"].iloc[0]
            carbon_savings = first_external_carbon_source - second_external_carbon_source
            carbon_savings_var.set(f"{carbon_savings:.4f}")

        run_count += 1

    except Exception as e:
        output_text.set(f"An error occurred during prediction:\n{e}")

units = {
    "Influent NH4+-N": "5.4 - 60.7 mg/L",
    "Influent TN": "9.8 - 108.0 mg/L",
    "TN of Zone 5": "4.8 - 44.6 mg/L",
    "TN of Zone 6": "3.0 - 42.3 mg/L",
    "TN of Zone 7": "2.8 - 38.9 mg/L",
    "Influent Flowrate": "0 - 69970.0 m³/d",
    "DO of Zone 5": "5.8 - 8.8 mg/L",
    "COD of Zone 6": "28.0 - 35.5 mg/L",
    "DO of Zone 7": "0.3 - 2.8 mg/L",
    "COD of Zone 7": "0 - 26.4 mg/L",
    "External Carbon Source": "1.9 - 2.3 m³/d"
}

root = tk.Tk()
root.title("Effluent TN Prediction and Optimization")
root.geometry("1200x800")
root.configure(bg="#e8f0f2")
default_font = ("Arial", 12)

main_frame = tk.Frame(root, bg="#e8f0f2", padx=20, pady=20)
main_frame.pack(expand=True)

title_label = tk.Label(main_frame, text="Effluent TN Prediction and Optimization", font=("Arial", 18, "bold"), bg="#e8f0f2")
title_label.grid(row=0, column=0, columnspan=8, pady=15)

# Left column labels and entry fields
left_labels = [
    ("Influent NH₄⁺-N", "Influent NH4+-N"),
    ("Influent TN", "Influent TN"),
    ("TN of Zone 5", "TN of Zone 5"),
    ("TN of Zone 6", "TN of Zone 6"),
    ("TN of Zone 7", "TN of Zone 7")
]

# Right column labels and entry fields
right_labels = [
    ("Influent Flowrate", "Influent Flowrate"),
    ("DO of Zone 5", "DO of Zone 5"),
    ("COD of Zone 6", "COD of Zone 6"),
    ("DO of Zone 7", "DO of Zone 7"),
    ("COD of Zone 7", "COD of Zone 7"),
    ("External Carbon Source", "External Carbon Source")
]

feature_entry_map = {}

# Left column entry fields
for i, (label, feature) in enumerate(left_labels, start=2):
    tk.Label(main_frame, text=label, font=default_font, bg="#e8f0f2").grid(row=i, column=0, padx=10, pady=5, sticky="e")
    entry = tk.Entry(main_frame, font=default_font, width=20)
    entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
    feature_entry_map[feature] = entry
    unit_text = units.get(feature, "")
    tk.Label(main_frame, text=f"({unit_text})", font=default_font, bg="#e8f0f2").grid(row=i, column=2, padx=10, pady=5, sticky="w")
tk.Label(main_frame, text="Water Quality Parameters", font=("Arial", 14, "bold"), bg="#e8f0f2").grid(row=1, column=1, columnspan=2, pady=10)

# Right column entry fields
for i, (label, feature) in enumerate(right_labels, start=2):
    tk.Label(main_frame, text=label, font=default_font, bg="#e8f0f2").grid(row=i, column=3, padx=10, pady=5, sticky="e")
    entry = tk.Entry(main_frame, font=default_font, width=20)
    entry.grid(row=i, column=4, padx=10, pady=5, sticky="ew")
    feature_entry_map[feature] = entry
    unit_text = units.get(feature, "")
    tk.Label(main_frame, text=f"({unit_text})", font=default_font, bg="#e8f0f2").grid(row=i, column=5, padx=10, pady=5, sticky="w")

tk.Label(main_frame, text="Controllable Parameters", font=("Arial", 14, "bold"), bg="#e8f0f2").grid(row=1, column=3, columnspan=2, pady=10)

original_tn = tk.StringVar()
original_power = tk.StringVar()
optimized_tn = tk.StringVar()
optimized_power = tk.StringVar()
energy_savings_var = tk.StringVar()
carbon_savings_var = tk.StringVar()
output_text = tk.StringVar()

# Display predict button
predict_button = tk.Button(main_frame, text="Predict", font=("Arial", 14), command=predict, bg="#4CAF50", fg="white", width=18)
predict_button.grid(row=8, column=0, columnspan=4, pady=20)

# Reset button
reset_button = tk.Button(main_frame, text="Reset", font=("Arial", 14), command=clear_inputs, bg="#FF5722", fg="white", width=18)
reset_button.grid(row=8, column=4, columnspan=4, pady=20)

# Result display frame
output_frame = tk.Frame(main_frame, bg="#e8f0f2")
output_frame.grid(row=9, column=0, columnspan=8, padx=10, pady=20)

tk.Label(output_frame, text="Original Effluent TN (mg/L):", font=default_font, bg="#e8f0f2").grid(row=0, column=0, sticky="e")
tk.Entry(output_frame, textvariable=original_tn, font=default_font, state="readonly").grid(row=0, column=1)

tk.Label(output_frame, text="Optimized Effluent TN (mg/L):", font=default_font, bg="#e8f0f2").grid(row=0, column=2, sticky="e")
tk.Entry(output_frame, textvariable=optimized_tn, font=default_font, state="readonly").grid(row=0, column=3)

tk.Label(output_frame, text="Original Energy Consumption (kWh):", font=default_font, bg="#e8f0f2").grid(row=1, column=0, sticky="e")
tk.Entry(output_frame, textvariable=original_power, font=default_font).grid(row=1, column=1)

tk.Label(output_frame, text="Optimized Energy Consumption (kWh):", font=default_font, bg="#e8f0f2").grid(row=1, column=2, sticky="e")
tk.Entry(output_frame, textvariable=optimized_power, font=default_font, state="readonly").grid(row=1, column=3)

tk.Label(output_frame, text="Energy Savings (kWh):", font=default_font, bg="#e8f0f2").grid(row=2, column=0, sticky="e")
tk.Entry(output_frame, textvariable=energy_savings_var, font=default_font, state="readonly").grid(row=2, column=1)

tk.Label(output_frame, text="Carbon Source Savings (m³/d):", font=default_font, bg="#e8f0f2").grid(row=2, column=2, sticky="e")
tk.Entry(output_frame, textvariable=carbon_savings_var, font=default_font, state="readonly").grid(row=2, column=3)

output_label = tk.Label(output_frame, textvariable=output_text, font=default_font, bg="#e8f0f2", fg="#FF5722")
output_label.grid(row=3, column=0, columnspan=4)

root.mainloop()

