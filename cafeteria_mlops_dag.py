from datetime import datetime
from pathlib import Path
import subprocess

from airflow.sdk import dag, task

BASE_DIR = Path("your_path")
PYTHON = BASE_DIR / ".venv" / "bin" / "python"


@dag(
    dag_id="cafeteria_mlops_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "mlflow", "cafeteria"]
)
def cafeteria_mlops_pipeline():

    @task
    def generate_data():
        subprocess.run(
            [
                str(PYTHON),
                str(BASE_DIR / "src" / "generate_data.py")
            ],
            check=True
        )

    @task
    def train_model():
        subprocess.run(
            [
                str(PYTHON),
                str(BASE_DIR / "src" / "train_model.py")
            ],
            check=True
        )

    @task
    def evaluate_model():
        subprocess.run(
            [
                str(PYTHON),
                str(BASE_DIR / "src" / "evaluate_model.py")
            ],
            check=True
        )

    data = generate_data()
    training = train_model()
    evaluation = evaluate_model()

    data >> training >> evaluation


cafeteria_mlops_pipeline()
