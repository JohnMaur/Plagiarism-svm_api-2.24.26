from flask import Flask, request, jsonify
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from deap import base, creator, tools
import random
from sklearn.datasets import make_classification

app = Flask(__name__)

# Synthetic dataset for GA optimization
X, y = make_classification(n_samples=200, n_features=50, n_informative=30, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# GA optimization (C, gamma)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox=base.Toolbox()
toolbox.register("attr_C", random.uniform,0.1,10)
toolbox.register("attr_gamma", random.uniform,0.0001,1)
toolbox.register("individual",tools.initCycle,creator.Individual,(toolbox.attr_C,toolbox.attr_gamma),n=1)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

def evaluate(ind):
    C,gamma=ind
    model=SVC(kernel='rbf',C=max(abs(C),0.0001),gamma=max(abs(gamma),0.0001))
    scores=cross_val_score(model,X_train,y_train,cv=3)
    return (scores.mean(),)
toolbox.register("evaluate",evaluate)
toolbox.register("mate",tools.cxBlend,alpha=0.5)
toolbox.register("mutate",tools.mutGaussian,mu=0,sigma=0.3,indpb=0.2)
toolbox.register("select",tools.selTournament,tournsize=3)

def ga_optimize():
    pop=toolbox.population(n=10)
    best_prev=0;stagnation=0
    mutation_rate=0.2
    for gen in range(10):
        offspring=toolbox.select(pop,len(pop))
        offspring=list(map(toolbox.clone,offspring))
        for c1,c2 in zip(offspring[::2],offspring[1::2]):
            if random.random()<0.7:
                toolbox.mate(c1,c2)
                del c1.fitness.values; del c2.fitness.values
        for mutant in offspring:
            if random.random()<mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid=[i for i in offspring if not i.fitness.valid]
        for ind,fit in zip(invalid,map(toolbox.evaluate,invalid)):
            ind.fitness.values=fit
        pop[:]=offspring
        best_fit=max(ind.fitness.values[0] for ind in pop)
        stagnation = stagnation+1 if best_fit<=best_prev else 0
        best_prev=max(best_prev,best_fit)
    best=tools.selBest(pop,1)[0]
    return max(abs(best[0]),0.0001), max(abs(best[1]),0.0001)

C_opt, gamma_opt = ga_optimize()
svm_model=SVC(kernel='rbf',C=C_opt,gamma=gamma_opt,probability=True)
svm_model.fit(X_train,y_train)

@app.route('/classify-text', methods=['POST'])
def classify_text():
    data = request.get_json()
    vector = np.array(data.get("vector")).reshape(1, -1)

    expected_features = X_train.shape[1]
    if vector.shape[1] != expected_features:
        adjusted = np.zeros((1, expected_features))
        adjusted[0, :min(expected_features, vector.shape[1])] = vector[0, :min(expected_features, vector.shape[1])]
        vector = adjusted

    # SVM margin distance
    decision = svm_model.decision_function(vector)[0]

    # Margin-based confidence computation
    confidence = min(abs(decision) * 100, 100)

    # Multi-stage decision logic
    if abs(decision) < 0.3:
        final_decision = "Uncertain – Requires Review"
    else:
        final_decision = "Plagiarized" if decision > 0 else "Original"

    return jsonify({
        "decision": final_decision,
        "confidence": round(confidence, 2),
        "marginDistance": round(float(decision), 4),
        "optimizedParameters": {
            "C": round(C_opt, 3),
            "gamma": round(gamma_opt, 4)
        }
    })

@app.route('/update-model', methods=['POST'])
def update_model():
    global X_train, y_train, svm_model

    data = request.get_json()
    X_new = np.array(data['X'])
    y_new = np.array(data['y'])

    expected_features = X_train.shape[1]

    # 🔧 FIX: Align feature dimensions
    if X_new.shape[1] != expected_features:
        adjusted = np.zeros((X_new.shape[0], expected_features))
        adjusted[:, :min(expected_features, X_new.shape[1])] = \
            X_new[:, :min(expected_features, X_new.shape[1])]
        X_new = adjusted

    # Append new samples safely
    X_train = np.vstack((X_train, X_new))
    y_train = np.concatenate((y_train, y_new))

    # Retrain SVM
    svm_model.fit(X_train, y_train)

    return jsonify({
        "message": "Model updated",
        "total_samples": int(len(X_train))
    })

if __name__=="__main__":
    app.run(port=5001)
