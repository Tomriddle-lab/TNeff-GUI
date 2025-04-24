import tkinter as tk
from tkinter import ttk
import pandas as pd
from nsga_optimization import optimize_parameters, predict_outputs, default_data
from topsis import topsis

# Unit data
units = {
    "Influent NH4+-N": "5.4 - 60.7 mg/L",
    "Influent TN": "9.8 - 108.0 mg/L",
    "TN of Zone 5": "4.8 - 44.6 mg/L",
    "TN of Zone 6": "3.0 - 42.3 mg/L",
    "TN of Zone 7": "2.8 - 38.9 mg/L",
    "Influent Flowrate": "0 - 69970.0 m³/d",
    "DO of Zone 5": "0.18 - 15.9 mg/L",
    "COD of Zone 6": "10.1 - 240.0 mg/L",
    "DO of Zone 7": "0.07 - 9.8 mg/L",
    "COD of Zone 7": "6.9 - 132.0 mg/L",
    "External Carbon Source": "0 - 15.3 m³/d"
}

# Suggested data
suggested = {
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


def validate_input(value):
    try:
        return float(value)
    except ValueError:
        return None


def clear_inputs():
    for entry in feature_entry_map.values():
        entry.delete(0, tk.END)
        entry.insert(0, default_data[feature_entry_map_inv[entry]])
    original_tn.set("")
    original_power.set("")
    optimized_tn.set("")
    optimized_power.set("")
    energy_savings_var.set("")
    carbon_savings_var.set("")
    output_text.set("")
    optimal_params_text.set("")


def predict_and_optimize():
    try:
        inputs = []
        for feature, entry in feature_entry_map.items():
            value = entry.get().strip()
            if not value:
                inputs.append(default_data[feature])
            else:
                validated = validate_input(value)
                if validated is None:
                    output_text.set(f"Invalid input for {feature}. Please enter a valid number.")
                    return
                inputs.append(validated)

        inputs_df = pd.DataFrame([inputs], columns=default_data.keys())
        inputs_scaled = scaler.transform(inputs_df)
        dmatrix = xgb.DMatrix(inputs_scaled)

        effluent_tn_pred = model_tn.predict(dmatrix)[0]
        power_consumption_pred = model_power.predict(dmatrix)[0]

        original_tn.set(f"{effluent_tn_pred:.4f}")
        original_power.set(f"{power_consumption_pred:.4f}")

        fixed_water_params = inputs_df[['Influent NH4+-N', 'Influent TN', 'Influent Flowrate',
                                        'TN of Zone 5', 'TN of Zone 6', 'TN of Zone 7']].values[0]
        initial_operating_params = inputs_df[['DO of Zone 5', 'COD of Zone 6', 'DO of Zone 7',
                                              'COD of Zone 7', 'External Carbon Source']].values[0]

        pareto_samples, true_energy, true_tn = optimize_parameters(
            fixed_water_params, initial_operating_params)

        pareto_data = np.array([(predict_outputs(np.concatenate([fixed_water_params, ind]))[0][0] - true_energy,
                                 predict_outputs(np.concatenate([fixed_water_params, ind]))[1][0] - true_tn)
                                for ind in pareto_samples])
        topsis_scores = topsis(pareto_data, weights=np.array([0.5, 0.5]))
        best_idx = np.argmax(topsis_scores)
        best_solution = list(pareto_samples[best_idx])

        energy_pred, tn_pred = predict_outputs(np.concatenate([fixed_water_params, best_solution]))

        optimized_tn.set(f"{tn_pred[0]:.4f}")
        optimized_power.set(f"{energy_pred[0]:.4f}")

        energy_savings = float(original_power.get()) - float(optimized_power.get())
        energy_savings_var.set(f"{energy_savings:.4f}")

        carbon_savings = float(inputs_df["External Carbon Source"].iloc[0]) - best_solution[4]
        carbon_savings_var.set(f"{carbon_savings:.4f}")

        optimal_params = {
            'DO of Zone 5': best_solution[0],
            'COD of Zone 6': best_solution[1],
            'DO of Zone 7': best_solution[2],
            'COD of Zone 7': best_solution[3],
            'External Carbon Source': best_solution[4]
        }
        params_list = [f"{k}: {v:.4f}" for k, v in optimal_params.items()]
        params_text = ""
        for i in range(0, len(params_list), 3):
            line = "                         ".join(params_list[i:i + 3]) + "\n"
            params_text += line
        while params_text.count('\n') < 3:
            params_text += '\n'
        optimal_params_text.set(params_text)

        output_text.set("Optimization completed successfully! The black text in the input boxes are examples of user input parameters, and the green text are the optimized parameters.")

        root.update_idletasks()
        req_height = main_frame.winfo_reqheight() + 20
        root.geometry(f"1350x{req_height}")

    except Exception as e:
        output_text.set(f"An error occurred during prediction/optimization:\n{str(e)}")


root = tk.Tk()
root.title("Effluent TN Prediction and Optimization")
root.geometry("1350x800")
root.configure(bg="#e8f0f2")
default_font = ("Arial", 12)

main_frame = tk.Frame(root, bg="#e8f0f2", padx=20, pady=20)
main_frame.pack(expand=True)

title_label = tk.Label(main_frame, text="Effluent TN Prediction and Optimization",
                       font=("Arial", 18, "bold"), bg="#e8f0f2")
title_label.grid(row=0, column=0, columnspan=8, pady=15)

left_labels = [
    ("Influent NH₄⁺-N", "Influent NH4+-N"),
    ("Influent TN", "Influent TN"),
    ("TN of Zone 5", "TN of Zone 5"),
    ("TN of Zone 6", "TN of Zone 6"),
    ("TN of Zone 7", "TN of Zone 7")
]

right_labels = [
    ("Influent Flowrate", "Influent Flowrate"),
    ("DO of Zone 5", "DO of Zone 5"),
    ("COD of Zone 6", "COD of Zone 6"),
    ("DO of Zone 7", "DO of Zone 7"),
    ("COD of Zone 7", "COD of Zone 7"),
    ("External Carbon Source", "External Carbon Source")
]

feature_entry_map = {}
feature_entry_map_inv = {}

for i, (label, feature) in enumerate(left_labels, start=2):
    tk.Label(main_frame, text=label, font=default_font, bg="#e8f0f2").grid(row=i, column=0, padx=10, pady=5, sticky="e")
    entry = tk.Entry(main_frame, font=default_font, width=10)
    entry.insert(0, default_data[feature])
    entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
    feature_entry_map[feature] = entry
    feature_entry_map_inv[entry] = feature
    unit_text = units.get(feature, "")
    tk.Label(main_frame, text=f"({unit_text})", font=default_font, bg="#e8f0f2").grid(row=i, column=2, padx=10, pady=5, sticky="w")

tk.Label(main_frame, text="Water Quality Parameters", font=("Arial", 14, "bold"), bg="#e8f0f2").grid(row=1, column=1, columnspan=2, pady=10)

for i, (label, feature) in enumerate(right_labels, start=2):
    tk.Label(main_frame, text=label, font=default_font, bg="#e8f0f2").grid(row=i, column=3, padx=10, pady=5, sticky="e")
    entry = tk.Entry(main_frame, font=default_font, width=10)
    entry.insert(0, default_data[feature])
    entry.grid(row=i, column=4, padx=10, pady=5, sticky="ew")
    feature_entry_map[feature] = entry
    feature_entry_map_inv[entry] = feature

    unit_text = units.get(feature, "")
    tk.Label(main_frame, text=f"({unit_text})", font=default_font, bg="#e8f0f2").grid(row=i, column=5, padx=10, pady=5, sticky="w")
    suggested_text = suggested.get(feature, "")
    tk.Label(main_frame, text=f"({suggested_text})", font=default_font, fg="#006400", bg="#e8f0f2").grid(row=i, column=6, padx=10, pady=5, sticky="w")

tk.Label(main_frame, text="Controllable Parameters", font=("Arial", 14, "bold"), bg="#e8f0f2").grid(row=1, column=3, columnspan=2, pady=10)
tk.Label(main_frame, text="Suggested Parameters", font=("Arial", 14, "bold"), bg="#e8f0f2", fg="#006400").grid(row=1, column=6, pady=10)

original_tn = tk.StringVar()
original_power = tk.StringVar()
optimized_tn = tk.StringVar()
optimized_power = tk.StringVar()
energy_savings_var = tk.StringVar()
carbon_savings_var = tk.StringVar()
output_text = tk.StringVar()
optimal_params_text = tk.StringVar()

tk.Button(main_frame, text="Predict and Optimize", font=("Arial", 14), command=predict_and_optimize,
          bg="#4CAF50", fg="white", width=18).grid(row=8, column=0, columnspan=4, pady=20)
tk.Button(main_frame, text="Reset", font=("Arial", 14), command=clear_inputs,
          bg="#FF5722", fg="white", width=18).grid(row=8, column=4, columnspan=4, pady=20)

output_frame = tk.Frame(main_frame, bg="#e8f0f2")
output_frame.grid(row=9, column=0, columnspan=8, padx=10, pady=20)

tk.Label(output_frame, text="Original Effluent TN (mg/L):", font=default_font, bg="#e8f0f2").grid(row=0, column=0, sticky="e")
tk.Entry(output_frame, textvariable=original_tn, font=default_font, state="readonly", width=10).grid(row=0, column=1)

tk.Label(output_frame, text="Optimized Effluent TN (mg/L):", font=default_font, bg="#e8f0f2").grid(row=0, column=2, sticky="e")
tk.Entry(output_frame, textvariable=optimized_tn, font=default_font, state="readonly", width=10).grid(row=0, column=3)

tk.Label(output_frame, text="Original Energy Consumption (kWh):", font=default_font, bg="#e8f0f2").grid(row=1, column=0, sticky="e")
tk.Entry(output_frame, textvariable=original_power, font=default_font, state="readonly", width=10).grid(row=1, column=1)

tk.Label(output_frame, text="Optimized Energy Consumption (kWh):", font=default_font, bg="#e8f0f2").grid(row=1, column=2, sticky="e")
tk.Entry(output_frame, textvariable=optimized_power, font=default_font, state="readonly", width=10).grid(row=1, column=3)

tk.Label(output_frame, text="Energy Savings (kWh):", font=default_font, bg="#e8f0f2").grid(row=2, column=0, sticky="e")
tk.Entry(output_frame, textvariable=energy_savings_var, font=default_font, state="readonly", width=10).grid(row=2, column=1)

tk.Label(output_frame, text="Carbon Source Savings (m³/d):", font=default_font, bg="#e8f0f2").grid(row=2, column=2, sticky="e")
tk.Entry(output_frame, textvariable=carbon_savings_var, font=default_font, state="readonly", width=10).grid(row=2, column=3)

tk.Label(output_frame, textvariable=output_text, font=default_font, bg="#e8f0f2", fg="#FF5722").grid(row=3, column=0, columnspan=4)

optimal_params_frame = tk.Frame(main_frame, bg="#e8f0f2", bd=2, relief="groove")
optimal_params_frame.grid(row=10, column=0, columnspan=8, padx=10, pady=10, sticky="nsew")

tk.Label(optimal_params_frame, text="Optimal Parameters Combination:", font=("Arial", 12, "bold"), bg="#e8f0f2", fg="#006400").pack()

optimal_params_display = tk.Text(optimal_params_frame, font=default_font, bg="#e8f0f2", wrap=tk.WORD, fg="#006400", height=3)
optimal_params_display.pack(pady=5, fill=tk.BOTH, expand=True)


def update_optimal_params_display(*args):
    params_text = optimal_params_text.get()
    optimal_params_display.delete(1.0, tk.END)
    optimal_params_display.insert(tk.END, params_text)


optimal_params_text.trace_add("write", update_optimal_params_display)

root.mainloop()
    