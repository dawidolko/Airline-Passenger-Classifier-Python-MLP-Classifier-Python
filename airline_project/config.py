"""
Configuration module for the Airline Passenger Satisfaction project.
Contains all constants: paths, column names, model hyperparameters,
grid-search parameters and data-split ratios.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "airline"
PLOTS_DIR = RESULTS_DIR / "plots"
TARGET_COLUMN = "Class"

CLASS_LABELS = ["Business", "Eco", "Eco Plus"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_LABELS)}

DROP_COLUMNS = ["Unnamed: 0", "id", "satisfaction"]
CATEGORICAL_COLUMNS = ["Gender", "Customer Type", "Type of Travel"]

NUMERIC_COLUMNS = [
    "Age",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

SERVICE_SCORE_COLUMNS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BASELINE_CONFIG = {
    "name": "baseline_MLP128_64",
    "hidden_sizes": (128, 64),
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 1024,
    "max_epochs": 35,
    "patience": 6,
    "min_delta": 1e-4,
}

ARCHITECTURES = [
    (128, 64),
    (256, 128, 64),
    (512, 256, 128),
]
DROPOUT_VALUES = [0.1, 0.2, 0.3]
LEARNING_RATES = [1e-3, 5e-4]
