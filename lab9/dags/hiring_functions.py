import os
from datetime import date
import pandas as pd
from sklearn.model_selection import train_test_split

# misma semilla para slit y para entrenamiento
seed = 123

def create_folders(**kwargs):
    date = kwargs.get("date")
    folder_name = f'{date.today()}'
    os.mkdir(folder_name)
    os.mkdir(f"{folder_name}/raw")
    os.mkdir(f"{folder_name}/splits")
    os.mkdir(f"{folder_name}/models")
    return

def split_data(**kwargs):
    dir = f"{kwargs.get('date')}"
    df = pd.read_csv(f"{dir}/raw/data_1.csv")
    target_col = "HiringDecision"
    X = df.drop(columns=[target_col])
    X_cols = X.columns
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    train_df = pd.DataFrame(X_train, columns=X_cols)
    train_df[target_col] = y_train

    test_df = pd.DataFrame(X_test, columns=X_cols)
    test_df[target_col] = y_test

    train_df.to_csv(f"{dir}/splits/train.csv")
    test_df.to_csv(f"{dir}/test.csv")

def preprocess_and_train(**kwargs):
    dir = f"{kwargs.get('date')}"
    train_df = pd.read_csv(f"{dir}/splits/train.csv")
    test_df = pd.read_csv(f"{dir}/splits/test.csv")