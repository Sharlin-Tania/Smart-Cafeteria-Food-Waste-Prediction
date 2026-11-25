import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# Project folder
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Create synthetic cafeteria data
n = 300

students = np.random.randint(100, 501, n)
meals_prepared = (students * np.random.uniform(1.05, 1.30, n)).astype(int)

weekend = np.random.randint(0, 2, n)
event_day = np.random.binomial(1, 0.15, n)
rainy_day = np.random.binomial(1, 0.30, n)
exam_day = np.random.binomial(1, 0.20, n)

previous_leftover = np.clip(
    np.random.normal(12, 5, n),
    0,
    30
)

# Create a waste-risk score
waste_score = (
    3 * (meals_prepared / students)
    + 1.2 * weekend
    + 1.5 * rainy_day
    + 1.0 * event_day
    + 1.2 * (previous_leftover / 10)
    - 0.8 * exam_day
    + np.random.normal(0, 0.3, n)
)

# HIGH = 1, LOW = 0
threshold = np.median(waste_score)
waste_risk = (waste_score > threshold).astype(int)

# Create dataframe
df = pd.DataFrame({
    "students": students,
    "meals_prepared": meals_prepared,
    "weekend": weekend,
    "event_day": event_day,
    "rainy_day": rainy_day,
    "exam_day": exam_day,
    "previous_leftover": previous_leftover.round(2),
    "waste_risk": waste_risk
})

# Save dataset
file_path = DATA_DIR / "cafeteria_data.csv"
df.to_csv(file_path, index=False)

print("Dataset created successfully!")
print(f"Saved to: {file_path}")
print(df.head())