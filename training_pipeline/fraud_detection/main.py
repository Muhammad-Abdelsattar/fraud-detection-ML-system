import os
import time

# Local module imports
import config
from data_preparation import prepare_data
from training import run_hpo_and_train
from evaluation import evaluate_model
from model_exporter import export_model

def run_pipeline():
    """Main function to orchestrate the ML pipeline."""
    
    start_time = time.time()
    
    # Ensure artifacts directory exists
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    
    # --- Step 1: Data Preparation ---
    print("--- Step 1/4: Preparing Data ---")
    X_train, X_test, y_train, y_test, features_dict = prepare_data()
    print("Data preparation complete.")
    
    # --- Step 2: HPO and Model Training ---
    print("\n--- Step 2/4: Running Hyperparameter Optimization and Training ---")
    final_pipeline = run_hpo_and_train(X_train, y_train, features_dict["num_cols"], features_dict["cat_cols"])
    print("Model training complete.")
    
    # --- Step 3: Model Evaluation ---
    print("\n--- Step 3/4: Evaluating Final Model ---")
    evaluate_model(final_pipeline, X_test, y_test)
    print("Evaluation complete.")

    # --- Step 4: Export Model for Inference ---
    print("\n--- Step 4/4: Exporting Model for Inference ---")
    X_sample = X_train.head(1)
    export_model(final_pipeline, X_sample, export_type=config.EXPORT_TYPE)
    
    end_time = time.time()
    print(f"\nPipeline finished successfully in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_pipeline()