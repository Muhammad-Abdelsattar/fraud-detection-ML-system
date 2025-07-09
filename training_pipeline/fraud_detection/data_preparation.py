from pathlib import Path
import pandas as pd
import numpy as np
from config import DATA_PATH, TARGET, COLS_TO_DROP, TEST_SIZE_RATIO


def read_data():
    data_path = Path(DATA_PATH)
    if data_path.is_file():
        return pd.read_csv(data_path)
    
    df_list = []
    for file in data_path.glob("*.csv"):
        temp_data = pd.read_csv(file)
        df_list.append(temp_data)
    return pd.concat(df_list, ignore_index=True)


def get_numerical_categorical_cols(data):
    # gets the numerical and categorical columns to be used in the preprocessor
    numerical_features = data.select_dtypes(include=np.number).columns.tolist()
    categorical_features = data.select_dtypes(exclude=np.number).columns.tolist()
    return {"num_cols":numerical_features, "cat_cols":categorical_features}

# def add_hour_of_day_column(df):
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df['hour_of_day'] = df['timestamp'].dt.hour
#     return df

def split_dataset_based_on_time(df):
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    features_df = df_sorted.drop(columns=[TARGET] + COLS_TO_DROP)
    target = df_sorted[TARGET]

    split_index = int(len(df_sorted) * (1 - TEST_SIZE_RATIO))
    X_train = features_df.iloc[:split_index]
    X_test = features_df.iloc[split_index:]
    y_train = target.iloc[:split_index]
    y_test = target.iloc[split_index:]

    return X_train, X_test, y_train, y_test

def prepare_data():
    """
    Loads, cleans, engineers features, and splits the data.
    Returns:
        X_train, X_test, y_train, y_test, numerical_features, categorical_features
    """
    print("Preparing data...")
    
    df = read_data()
    
    # Adds the hour of day feature from the timestamp
    # df = add_hour_of_day_column(df)
    
    # Convertin boolean features to integers
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)

    # Time-based Split
    X_train, X_test, y_train, y_test = split_dataset_based_on_time(df)

    features_names_dict = get_numerical_categorical_cols(X_train)

    print("Data preparation complete.")
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, features_names_dict