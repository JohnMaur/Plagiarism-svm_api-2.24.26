# from flask import Flask, request, jsonify
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, cross_val_score
# import numpy as np
# import random
# from deap import base, creator, tools, algorithms
# from sklearn.datasets import make_classification

# # ==============================
# # 1️⃣  Synthetic Dataset for GA Optimization
# # ==============================
# X, y = make_classification(
#     n_samples=200,
#     n_features=50,
#     n_informative=30,
#     n_redundant=10,
#     n_classes=2,
#     random_state=42
# )

# # Split data for training and testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Placeholder vectorizer (for later actual text input)
# vectorizer = TfidfVectorizer()

# # ==============================
# # 2️⃣  Genetic Algorithm Setup
# # ==============================
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize accuracy
# creator.create("Individual", list, fitness=creator.FitnessMax)

# toolbox = base.Toolbox()

# # Define valid ranges for C and gamma
# toolbox.register("attr_C", random.uniform, 0.1, 10.0)
# toolbox.register("attr_gamma", random.uniform, 0.0001, 1.0)
# toolbox.register("individual", tools.initCycle, creator.Individual,
#                  (toolbox.attr_C, toolbox.attr_gamma), n=1)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# def evaluate(individual):
#     """Evaluate an individual's fitness (mean accuracy)."""
#     C, gamma = individual

#     # Ensure valid positive values
#     C = abs(float(C))
#     if C < 0.0001:
#         C = 0.0001

#     gamma = abs(float(gamma))
#     if gamma < 0.0001:
#         gamma = 0.0001

#     model = SVC(C=C, gamma=gamma, kernel='rbf')
#     scores = cross_val_score(model, X_train, y_train, cv=3)
#     return (scores.mean(),)


# toolbox.register("evaluate", evaluate)
# toolbox.register("mate", tools.cxBlend, alpha=0.5)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2)
# toolbox.register("select", tools.selTournament, tournsize=3)

# # ==============================
# # 3️⃣  GA Optimization Loop
# # ==============================
# def ga_optimize():
#     population = toolbox.population(n=10)
#     NGEN = 10
#     mutation_rate = 0.2
#     stagnation = 0
#     best_prev = 0

#     for gen in range(NGEN):
#         offspring = toolbox.select(population, len(population))
#         offspring = list(map(toolbox.clone, offspring))

#         # Crossover
#         for child1, child2 in zip(offspring[::2], offspring[1::2]):
#             if random.random() < 0.7:
#                 toolbox.mate(child1, child2)
#                 del child1.fitness.values, child2.fitness.values

#         # Mutation
#         for mutant in offspring:
#             if random.random() < mutation_rate:
#                 toolbox.mutate(mutant)
#                 del mutant.fitness.values

#         # Evaluate new population
#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#         fitnesses = list(map(toolbox.evaluate, invalid_ind))
#         for ind, fit in zip(invalid_ind, fitnesses):
#             ind.fitness.values = fit

#         population[:] = offspring
#         fits = [ind.fitness.values[0] for ind in population]
#         best_fit = max(fits)

#         if best_fit <= best_prev:
#             stagnation += 1
#         else:
#             stagnation = 0
#             best_prev = best_fit

#         # Adjust mutation rate dynamically
#         if stagnation >= 2:
#             mutation_rate = min(0.8, mutation_rate + 0.1)
#         else:
#             mutation_rate = max(0.1, mutation_rate - 0.05)

#         print(f"Gen {gen+1}/{NGEN} | Best Fitness: {best_fit:.4f} | Mutation Rate: {mutation_rate:.2f}")

#     best_individual = tools.selBest(population, 1)[0]
#     best_C, best_gamma = best_individual
#     best_C = max(0.0001, abs(best_C))
#     best_gamma = max(0.0001, abs(best_gamma))

#     print(f"\n✅ Best parameters found: C={best_C:.3f}, gamma={best_gamma:.4f}")
#     return best_C, best_gamma


# # Run optimization
# C_opt, gamma_opt = ga_optimize()

# # ==============================
# # 4️⃣  Train Final Optimized SVM
# # ==============================
# svm_model = SVC(kernel='rbf', C=C_opt, gamma=gamma_opt, probability=True)
# svm_model.fit(X_train, y_train)


# # ==============================
# # 5️⃣  Flask API
# # ==============================
# app = Flask(__name__)

# @app.route("/classify-text", methods=["POST"])
# def classify_text():
#     try:
#         data = request.get_json()
#         input_vector = data.get("vector")

#         if not input_vector or not isinstance(input_vector, list):
#             return jsonify({"error": "No valid TF-IDF vector provided"}), 400

#         # Convert vector to numpy array and reshape for sklearn
#         vector_np = np.array(input_vector).reshape(1, -1)

#         # If your Node vector length doesn't match training features, fix mismatch
#         expected_features = X_train.shape[1]
#         if vector_np.shape[1] != expected_features:
#             # Adjust vector length (pad or truncate)
#             adjusted = np.zeros((1, expected_features))
#             min_len = min(expected_features, vector_np.shape[1])
#             adjusted[0, :min_len] = vector_np[0, :min_len]
#             vector_np = adjusted

#         # Predict using trained and GA-optimized SVM
#         prediction = svm_model.predict(vector_np)[0]
#         probability = svm_model.predict_proba(vector_np)[0][1] * 100

#         return jsonify({
#             "isPlagiarized": bool(prediction),
#             "confidence": round(probability, 2),
#             "optimizedParameters": {
#                 "C": round(C_opt, 3),
#                 "gamma": round(gamma_opt, 4)
#             }
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# @app.route("/compare-semantic", methods=["POST"])
# def compare_semantic():
#     try:
#         data = request.get_json()
#         text_vector = np.array(data.get("textVector")).reshape(1, -1)
#         snippet_vectors = np.array(data.get("snippetVectors"))

#         results = []

#         expected_features = X_train.shape[1]

#         # Adjust text_vector length if needed
#         if text_vector.shape[1] != expected_features:
#             adjusted_text = np.zeros((1, expected_features))
#             min_len = min(expected_features, text_vector.shape[1])
#             adjusted_text[0, :min_len] = text_vector[0, :min_len]
#             text_vector = adjusted_text

#         for snippet_vector in snippet_vectors:
#             v = np.array(snippet_vector).reshape(1, -1)

#             # Adjust snippet vector
#             if v.shape[1] != expected_features:
#                 adjusted_v = np.zeros((1, expected_features))
#                 min_len = min(expected_features, v.shape[1])
#                 adjusted_v[0, :min_len] = v[0, :min_len]
#                 v = adjusted_v

#             # 🔹 Combine text and snippet vector to form a "difference" vector
#             combined_vector = np.abs(text_vector - v)

#             # Predict similarity using SVM (probability)
#             prob = svm_model.predict_proba(combined_vector)[0][1] * 100

#             results.append({
#                 "semanticSimilarity": round(prob, 2),
#                 "confidence": round(prob, 2)
#             })

#         return jsonify({"semanticResults": results})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(port=5001)


# New lines added 1/25/26
from flask import Flask, request, jsonify
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.datasets import make_classification
from deap import base, creator, tools

# =====================================================
# SOP 1 – FEATURE SPACE EXPANSION
# =====================================================
vectorizer = FeatureUnion([
    ("unigram", TfidfVectorizer(ngram_range=(1, 1), max_features=3000)),
    ("bigram", TfidfVectorizer(ngram_range=(2, 2), max_features=2000)),
])

# Synthetic dataset placeholder
X_raw, y = make_classification(
    n_samples=200,
    n_features=50,
    n_informative=30,
    n_redundant=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.3, random_state=42
)

# =====================================================
# GENETIC ALGORITHM FOR SVM PARAMETER OPTIMIZATION
# =====================================================
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_C", random.uniform, 0.1, 10)
toolbox.register("attr_gamma", random.uniform, 0.0001, 1)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_C, toolbox.attr_gamma), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    C, gamma = individual
    C, gamma = max(abs(C), 0.0001), max(abs(gamma), 0.0001)
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    scores = cross_val_score(model, X_train, y_train, cv=3)
    return (scores.mean(),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def ga_optimize():
    population = toolbox.population(n=10)
    mutation_rate = 0.2
    best_prev = 0
    stagnation = 0

    for gen in range(10):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate
        invalid = [i for i in offspring if not i.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = offspring
        best_fit = max(ind.fitness.values[0] for ind in population)

        stagnation = stagnation + 1 if best_fit <= best_prev else 0
        best_prev = max(best_prev, best_fit)

        mutation_rate = min(0.8, mutation_rate + 0.1) if stagnation >= 2 else max(0.1, mutation_rate - 0.05)
        print(f"Gen {gen+1} | Best Accuracy: {best_fit:.4f}")

    best = tools.selBest(population, 1)[0]
    return max(abs(best[0]), 0.0001), max(abs(best[1]), 0.0001)

C_opt, gamma_opt = ga_optimize()

# =====================================================
# SOP 3 – DYNAMIC TRAINING
# =====================================================
svm_model = SVC(kernel="rbf", C=C_opt, gamma=gamma_opt, probability=True)
svm_model.fit(X_train, y_train)

def retrain_svm(new_X, new_y):
    global svm_model
    svm_model.fit(new_X, new_y)

# =====================================================
# FLASK API
# =====================================================
app = Flask(__name__)

# Margin-aware multi-stage decision
@app.route("/classify-text", methods=["POST"])
def classify_text():
    try:
        data = request.get_json()
        vector = np.array(data.get("vector")).reshape(1, -1)
        expected = X_train.shape[1]
        if vector.shape[1] != expected:
            adjusted = np.zeros((1, expected))
            adjusted[0, :min(expected, vector.shape[1])] = vector[0, :min(expected, vector.shape[1])]
            vector = adjusted

        decision = svm_model.decision_function(vector)[0]
        probability = svm_model.predict_proba(vector)[0][1] * 100

        if abs(decision) < 0.3:
            final_decision = "Uncertain – Requires Review"
        else:
            final_decision = "Plagiarized" if decision > 0 else "Original"

        return jsonify({
            "decision": final_decision,
            "confidence": round(probability, 2),
            "marginDistance": round(float(decision), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to update model dynamically
@app.route("/update-model", methods=["POST"])
def update_model():
    try:
        data = request.get_json()
        new_X = np.array(data.get("vectors"))
        new_y = np.array(data.get("labels"))
        retrain_svm(new_X, new_y)
        return jsonify({"message": "SVM model retrained successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001)


@app.route('/update-model', methods=['POST'])
def update_model():
    data = request.get_json()
    X_new = np.array(data['X'])
    y_new = np.array(data['y'])
    svm_model.fit(X_new, y_new)
    return jsonify({"message": "Model updated"})