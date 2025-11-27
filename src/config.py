import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, FIGURES_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Model Hyperparameters
XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'early_stopping_rounds': 50,
    'objective': 'reg:squarederror',
    'enable_categorical': True
}

# Feature Lists
TARGET = 'PJME_MW'
LAG_HOURS = [24, 48, 72, 96, 168] # This was missing/undefined