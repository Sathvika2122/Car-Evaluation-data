import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import random

# Load and encode dataset
df = pd.read_csv('car.data', header=None)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Genetic Algorithm for RF Hyperparameters ---

# Define search space
def random_hyperparams():
    return {
        'n_estimators': random.choice([50, 100, 150, 200]),
        'max_depth': random.choice([None, 5, 10, 15, 20]),
        'min_samples_split': random.choice([2, 5, 10]),
        'min_samples_leaf': random.choice([1, 2, 4])
    }

# Fitness function
def fitness(params):
    model = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=3)
    return scores.mean()

# GA setup
POP_SIZE = 15
N_GEN = 100
MUTATION_RATE = 0.2

population = [random_hyperparams() for _ in range(POP_SIZE)]

for gen in range(N_GEN):
    scored_pop = [(individual, fitness(individual)) for individual in population]
    scored_pop.sort(key=lambda x: x[1], reverse=True)
    print(f"Generation {gen + 1} Best Score: {scored_pop[0][1]:.4f}")

    next_gen = [x[0] for x in scored_pop[:2]]  # Elitism: top 2

    # Generate new offspring
    while len(next_gen) < POP_SIZE:
        parent1, parent2 = random.sample(scored_pop[:5], 2)
        child = {}
        for key in parent1[0]:
            child[key] = random.choice([parent1[0][key], parent2[0][key]])
            # Mutation
            if random.random() < MUTATION_RATE:
                child[key] = random_hyperparams()[key]
        next_gen.append(child)

    population = next_gen

# Train best model
best_params = scored_pop[0][0]
print("\n Best Parameters from GA:", best_params)

rf_ga = RandomForestClassifier(**best_params, random_state=42)
rf_ga.fit(X_train, y_train)
y_pred_rf = rf_ga.predict(X_test)

print("\n GA-tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\n Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoders['class'].classes_,
            yticklabels=label_encoders['class'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('GA-Tuned Random Forest Confusion Matrix')
plt.tight_layout()
plt.show()

# Save model
joblib.dump(rf_ga, 'ga_rf_car_model.pkl')
print(" Model saved as 'ga_rf_car_model.pkl'")
