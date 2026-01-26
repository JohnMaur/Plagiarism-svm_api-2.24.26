from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset for training 
documents = [
    "This is an original text document.",
    "This document is an original piece of content.",
    "This is plagiarized content taken from another source.",
    "Plagiarized content often mirrors other works exactly."
]
labels = [0, 0, 1, 1]  # 0 = original, 1 = plagiarized

# Train SVM Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Flask App Setup
app = Flask(__name__)

@app.route("/check-plagiarism", methods=["POST"])
def check_plagiarism():
    try:
        data = request.json
        content_to_check = data.get("text", "")
        
        if not content_to_check.strip():
            return jsonify({"error": "No content provided"}), 400
        
        # Transform input text to TF-IDF vector
        input_vector = vectorizer.transform([content_to_check]).toarray()
        
        # Predict plagiarism probability
        prediction = svm_model.predict(input_vector)[0]
        probability = svm_model.predict_proba(input_vector)[0][1] * 100  # Plagiarism confidence
        
        response = {
            "isPlagiarized": bool(prediction),
            "confidence": probability
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001)


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

# if __name__ == "__main__":
#     app.run(port=5001)
