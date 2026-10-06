
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the labeled student dataset from CSV
DATA_PATH = Path(__file__).with_name("data") / "students.csv"
FEATURES = [
    "study_hours",
    "previous_marks",
    "attendance",
    "assignments_completed",
    "sleep_hours",
]

df = pd.read_csv(DATA_PATH)

# 2. Clean data
df = df.drop_duplicates()
df = df.dropna(subset=["passed"])

# 3. Separate features and target
X = df[FEATURES]
y = df["passed"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 5. Define candidate models
models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(max_depth=3, random_state=42)
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )
    )
}

# 6. Compare using training data and validation
# For this tiny demo, use cross-validation on training data.
from sklearn.model_selection import cross_val_score

results = {}

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train,
        cv=3, scoring="accuracy"
    )
    results[name] = scores.mean()
    print(name, "CV accuracy:", round(scores.mean(), 3))

# 7. Select model using cross-validation accuracy
best_name = max(results, key=results.get)
best_model = models[best_name]

print("\nSelected model:", best_name)

# 8. Fit selected model on training data
best_model.fit(X_train, y_train)

# 9. Final test evaluation
predictions = best_model.predict(X_test)

print("Test accuracy:",
      accuracy_score(y_test, predictions))

print(classification_report(
    y_test, predictions,
    zero_division=0
))

# 10. Save trained pipeline
joblib.dump(best_model, "student_model.joblib")
print("Model saved!")