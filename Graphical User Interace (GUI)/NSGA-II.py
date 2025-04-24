import numpy as np
import random
from deap import base, creator, tools, algorithms
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

# Variable ranges
variable_ranges = [
    (5.8, 7.5),  # DO of Zone 5
    (28.0, 35.5),  # COD of Zone 6
    (0.3, 2.8),  # DO of Zone 7
    (1.0, 26.4),  # COD of Zone 7
    (1.9, 2.3)  # External Carbon Source
]

df = pd.DataFrame([default_data])
scaler = StandardScaler()
scaler.fit(df)

model_tn = xgb.Booster()
model_power = xgb.Booster()

prediction_cache = {}


def predict_outputs(data):
    if data.ndim == 1:
        key = tuple(data)
        data = data.reshape(1, -1)
    else:
        return model_power.predict(xgb.DMatrix(data)), model_tn.predict(xgb.DMatrix(data))

    if key in prediction_cache:
        return np.array([prediction_cache[key][0]]), np.array([prediction_cache[key][1]])

    dmatrix = xgb.DMatrix(data)
    pred_energy = model_power.predict(dmatrix)
    pred_tn = model_tn.predict(dmatrix)
    prediction_cache[key] = (pred_energy[0], pred_tn[0])
    return pred_energy, pred_tn


def evaluate(individual, fixed_water_params, true_energy, true_tn):
    full_input = np.concatenate([fixed_water_params, individual])
    energy, tn = predict_outputs(np.array(full_input))
    delta_energy = energy[0] - true_energy
    delta_tn = tn[0] - true_tn
    return delta_energy, delta_tn


def optimize_parameters(fixed_water_params, initial_operating_params):
    if not hasattr(creator, 'FitnessMin'):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: [random.uniform(r[0], r[1]) for r in variable_ranges])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[r[0] for r in variable_ranges],
                     up=[r[1] for r in variable_ranges], eta=15.0)

    def custom_mutate(individual):
        individual = tools.mutPolynomialBounded(individual, low=[r[0] for r in variable_ranges],
                                                up=[r[1] for r in variable_ranges], eta=20.0, indpb=0.3)[0]
        for i in range(len(individual)):
            low, up = variable_ranges[i]
            individual[i] = max(low, min(up, individual[i]))
            if isinstance(individual[i], complex) or np.isnan(individual[i]) or np.isinf(individual[i]):
                individual[i] = (low + up) / 2
        return individual,

    toolbox.register("mutate", custom_mutate)
    toolbox.register("select", tools.selNSGA2)

    full_input = np.concatenate([fixed_water_params, initial_operating_params])
    true_energy, true_tn = predict_outputs(full_input)

    def fitness_func(ind):
        return evaluate(ind, fixed_water_params, true_energy[0], true_tn[0])

    toolbox.register("evaluate", fitness_func)

    pop = toolbox.population(n=100)
    pop[0][:] = initial_operating_params.copy()
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    hof = tools.ParetoFront()
    hof.update(pop)
    best_fitness = min([ind.fitness.values[0] for ind in pop])
    stall_count = 0

    for gen in range(20):
        offspring = algorithms.varOr(pop, toolbox, lambda_=100, cxpb=0.5, mutpb=0.5)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + offspring, k=20)
        hof.update(pop)
        current_best = min([ind.fitness.values[0] for ind in pop])
        if current_best < best_fitness * 0.99:
            best_fitness = current_best
            stall_count = 0
        else:
            stall_count += 1
        if stall_count >= 5:
            break

    n_samples = min(60, len(hof))
    pareto_samples = [hof[i] for i in np.linspace(0, len(hof) - 1, n_samples, dtype=int)]
    return pareto_samples, true_energy[0], true_tn[0]
    