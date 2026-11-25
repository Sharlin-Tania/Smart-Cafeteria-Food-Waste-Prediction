from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "cafeteria_data.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Cafeteria_Food_Waste_Risk")

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Start MLflow run
with mlflow.start_run() as run:

    model.fit(X_train, y_train)

    # Training accuracy
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    # Test accuracy
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    # Log parameters
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("test_size", 0.2)

    # Log metrics
    mlflow.log_metric(
        "train_accuracy",
        train_accuracy
    )

    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )

    # Log model
    mlflow.sklearn.log_model(
    model,
    "model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("Model training completed!")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"MLflow Run ID: {run.info.run_id}")

    # Save run ID
    run_id_file = REPORT_DIR / "run_id.txt"

    with open(run_id_file, "w") as f:
        f.write(run.info.run_id)