from pathlib import Path

import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "cafeteria_data.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

features = [
    "students",
    "meals_prepared",
    "weekend",
    "event_day",
    "rainy_day",
    "exam_day",
    "previous_leftover"
]

X = df[features]
y = df["waste_risk"]

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train a model again for evaluation
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

# Save evaluation report
report = classification_report(
    y_test,
    predictions
)

report_file = REPORT_DIR / "evaluation_report.txt"

with open(report_file, "w") as f:
    f.write("CAFETERIA FOOD WASTE MODEL EVALUATION\n")
    f.write("=" * 45 + "\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("Evaluation completed!")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Report saved to: {report_file}")