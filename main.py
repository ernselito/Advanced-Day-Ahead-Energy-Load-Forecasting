from src import data, features, training, visualization
import pandas as pd

def main():
    # 1. Get Data
    raw_data_path = data.download_data()
    df = data.load_and_clean_data(raw_data_path)

    # 2. Create Features
    df_processed = features.generate_features(df)

    # 3. Train Model
    model = training.train_model(df_processed)

    # 4. Save Artifacts
    training.save_model(model)
    
    # 5. Visualization & Saving Figures
    # We need to generate predictions on a test set to visualize. 
    # For simplicity in this main script, we'll just use the last portion of data as a proxy for "test" 
    # to demonstrate the saving functionality. In a real pipeline, you'd have a separate evaluation step.
    
    # ... (Split logic similar to training.py to get X_test and y_test for visualization) ...
    # For demonstration, let's just grab the last 20% of processed data
    split_idx = int(len(df_processed) * 0.8)
    test_df = df_processed.iloc[split_idx:].copy()
    X_test = test_df.drop(columns=['PJME_MW'])
    y_test = test_df['PJME_MW']
    
    test_df['Actual'] = y_test
    test_df['Prediction'] = model.predict(X_test)
    
    # Generate and save plots
    visualization.plot_forecast(test_df, week_start_date=test_df.index.min())
    visualization.plot_shap_summary(model, X_test)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()