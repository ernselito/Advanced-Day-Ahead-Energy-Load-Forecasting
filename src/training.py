import xgboost as xgb
import joblib
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from src import config

def train_model(df):
    """
    Trains XGBoost using Time Series Cross Validation.
    Returns the trained model and metrics.
    """
    print("Starting training...")
    
    # Prepare Features (X) and Target (y)
    features = [col for col in df.columns if col != config.TARGET]
    X = df[features]
    y = df[config.TARGET]

    tscv = TimeSeriesSplit(n_splits=5, test_size=24*365)
    fold = 0
    model = None
    
    for train_index, test_index in tscv.split(X):
        fold += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Init model with params from config
        model = xgb.XGBRegressor(**config.XGB_PARAMS)

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )

        # Basic evaluation for the fold
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        print(f"Fold {fold} RMSE: {rmse:.2f} MW")

    return model

def save_model(model, filename='xgboost_model.pkl'):
    """Saves model to the models folder."""
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
        
    filepath = os.path.join(config.MODEL_DIR, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")