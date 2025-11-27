import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import pandas as pd
from src import config

def plot_forecast(df, week_start_date=None):
    """
    Plots actual vs predicted load for a specific week and saves to figures folder.
    """
    if week_start_date is None:
        week_start_date = df.index.min()
        
    week_end_date = week_start_date + pd.Timedelta(days=7)
    subset = df.loc[(df.index >= week_start_date) & (df.index < week_end_date)]

    plt.figure(figsize=(15, 5))
    subset['Actual'].plot(ax=plt.gca(), label='Actual Load', title='Forecast vs Actuals (1 Week Zoom)')
    subset['Prediction'].plot(ax=plt.gca(), label='Model Forecast', style='--')
    plt.legend()
    
    # Save figure
    save_path = os.path.join(config.FIGURES_DIR, 'forecast_plot.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Forecast plot saved to {save_path}")
    plt.close() # Close plot to free memory

def plot_shap_summary(model, X_test):
    """
    Generates and saves SHAP summary plot.
    """
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    plt.title("Feature Importance: What drives Energy Demand?")
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    
    # Save figure
    save_path = os.path.join(config.FIGURES_DIR, 'feature_importance_plot.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"SHAP plot saved to {save_path}")
    plt.close()