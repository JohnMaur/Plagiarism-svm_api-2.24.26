from flask import Flask, request, jsonify
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from deap import base, creator, tools
import random
from sklearn.datasets import make_classification
import os

app = Flask(__name__)

# -------------------------------
# Synthetic dataset (bootstrap)
# -------------------------------
X, y = make_classification(
    n_samples=200,
    n_features=50,
    n_informative=30,
    n_redundant=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Genetic Algorithm Optimization
# -------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_C", random.uniform, 0.1, 10)
toolbox.register("attr_gamma", random.uniform, 0.0001, 1)
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.attr_C, toolbox.attr_gamma),
    n=1
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    C, gamma = ind
    model = SVC(
        kernel="rbf",
        C=max(abs(C), 0.0001),
        gamma=max(abs(gamma), 0.0001)
    )
    scores = cross_val_score(model, X_train, y_train, cv=3)
    return (scores.mean(),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def ga_optimize():
    pop = toolbox.population(n=10)
    best_prev = 0
    stagnation = 0

    for _ in range(10):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [i for i in offspring if not i.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = offspring
        best = max(ind.fitness.values[0] for ind in pop)
        stagnation = stagnation + 1 if best <= best_prev else 0
        best_prev = max(best_prev, best)

    best = tools.selBest(pop, 1)[0]
    return max(abs(best[0]), 0.0001), max(abs(best[1]), 0.0001)

C_opt, gamma_opt = ga_optimize()

svm_model = SVC(
    kernel="rbf",
    C=C_opt,
    gamma=gamma_opt,
    probability=True
)
svm_model.fit(X_train, y_train)

# -------------------------------
# Classify text (NO confidence here)
# -------------------------------
@app.route("/classify-text", methods=["POST"])
def classify_text():
    data = request.get_json()
    vector = np.array(data["vector"]).reshape(1, -1)

    expected = X_train.shape[1]
    if vector.shape[1] != expected:
        padded = np.zeros((1, expected))
        padded[0, :min(expected, vector.shape[1])] = vector[0, :min(expected, vector.shape[1])]
        vector = padded

    prob = svm_model.predict_proba(vector)[0][1]
    decision = "Plagiarized" if prob >= 0.5 else "Original"

    return jsonify({
        "decision": decision,
        "svmProbability": round(prob * 100, 2)
    })

# -------------------------------
# Update model incrementally
# -------------------------------
@app.route("/update-model", methods=["POST"])
def update_model():
    global X_train, y_train

    data = request.get_json()
    X_new = np.array(data["X"])
    y_new = np.array(data["y"])

    expected = X_train.shape[1]
    if X_new.shape[1] != expected:
        padded = np.zeros((X_new.shape[0], expected))
        padded[:, :min(expected, X_new.shape[1])] = X_new[:, :min(expected, X_new.shape[1])]
        X_new = padded

    X_train = np.vstack((X_train, X_new))
    y_train = np.concatenate((y_train, y_new))

    svm_model.fit(X_train, y_train)

    return jsonify({
        "message": "Model updated",
        "samples": int(len(X_train))
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
