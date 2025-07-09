DATA_PATH = "/teamspace/studios/this_studio/data_generation/final_project/synthetic_fraud_data_v3.csv"
MODELS_PATH = "/gcs/fraud-detection-machine-learning-artifacts/models"
TEST_SIZE_RATIO = 0.2

TARGET = 'is_flagged_fraud'

COLS_TO_DROP = [
    'timestamp', 'user_id', 'card_id', 'merchant_id', 
    'ip_address', 'device_id', 'transaction_id', 'fraud_scenario'
]

# For Optuna Hyperparameter Optimization
STUDY_NAME = "xgboost-fraud-detection"
N_TRIALS = 3 # Number of HPO trials to run
N_SPLITS_CV = 3 # Number of folds for cross-validation in HPO

# Artifacts
ARTIFACTS_DIR = "/teamspace/studios/this_studio/final_project/fraud_detection/training_pipeline/artifacts"
EXPORT_TYPE = "joblib"
MODEL_OUTPUT_PATH_ONNX = f"{ARTIFACTS_DIR}/model.onnx"
MODEL_OUTPUT_PATH_JOBLIB = f"{ARTIFACTS_DIR}/model.joblib"
CONFUSION_MATRIX_PATH = f"{ARTIFACTS_DIR}/confusion_matrix.png"