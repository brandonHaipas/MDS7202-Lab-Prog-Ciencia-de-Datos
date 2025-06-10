import os
import glob
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Seed Aleatoria
seed = 123

# Variable home
home_dir = os.getenv('AIRFLOW_HOME')

def create_folders(**kwargs):
    date = kwargs.get("ds")
    os.mkdir(date)
    os.mkdir(f"{home_dir}/{date}/raw")
    os.mkdir(f"{home_dir}/{date}/splits")
    os.mkdir(f"{home_dir}/{date}/models")
    os.mkdir(f"{home_dir}/{date}/preprocessed")
    return

def load_and_merge(**kwargs):
    dir = f"{kwargs.get('ds')}"
    df_1 = pd.read_csv(f"{home_dir}/{dir}/raw/data_1.csv")
    df_2 = None
    new_df = None
    try:
        df_2 = pd.read_csv(f"{home_dir}/{dir}/raw/data_2.csv")
    except FileNotFoundError:
        print("No data_2.csv file found, proceeding without it")
    if df_2 is None:
        new_df = df_1
    else:
        df_2 = df_2[df_1.columns.to_list()] # mismo orden de columnas
        new_df = pd.concat([df_1, df_2], ignore_index=True)
    new_df.to_csv(f"{home_dir}/{dir}/preprocessed/prep_data.csv")
    return

def split_data(**kwargs):
    dir = f"{kwargs.get('ds')}"
    df = pd.read_csv(f"{home_dir}/{dir}/preprocessed/prep_data.csv")
    target_col = "HiringDecision"
    X = df.drop(columns=[target_col])
    X_cols = X.columns
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    train_df = pd.DataFrame(X_train, columns=X_cols)
    train_df[target_col] = y_train

    test_df = pd.DataFrame(X_test, columns=X_cols)
    test_df[target_col] = y_test

    train_df.to_csv(f"{home_dir}/{dir}/splits/train.csv", index=False)
    test_df.to_csv(f"{home_dir}/{dir}/splits/test.csv", index=False)

class TypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type_to_transform):
        self.type_to_transform = type_to_transform
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.astype(self.type_to_transform)
    
    def set_output(self, transform=None):
        self._transform_output = transform
        return self

def train_model(model,model_name, **kwargs):
    dir = f"{kwargs.get('ds')}"
    train_df = pd.read_csv(f"{home_dir}/{dir}/splits/train.csv")

    # basandose en data_1_report, se puede concluir que las unicas variables que necesitan one_hot_encoding son EducationLevel RecruitmentStrategy.
    # el resto de columnas "categoricas" solo necesita un cambio de tipos.
    # para el resto de variables se podría aplicar un scaler, dado que no hay un mayor desbalance se puede usar directamente un minmax scaler.

    encode_cols = ["EducationLevel", "RecruitmentStrategy"]
    astype_cols = ["PreviousCompanies"]
    minmax_cols = ["Age", "ExperienceYears", "PreviousCompanies", "DistanceFromCompany", "InterviewScore", "SkillScore"]

    encoder = OneHotEncoder(sparse_output=False)
    scaler = MinMaxScaler(feature_range=(0,1))
    type_transformer_int = TypeTransformer(type_to_transform=int)
    type_transformer_str = TypeTransformer(type_to_transform=str)

    type_transformer = ColumnTransformer([
        ("Type Transformer Categorical", type_transformer_str, encode_cols),
        ("Type Transformer Numerical", type_transformer_int, astype_cols),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False)

    encoder_transformer = ColumnTransformer([
        ("One Hot Encoding", encoder, encode_cols),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False)

    scaler_transformer = ColumnTransformer([
        ("MinMax Scaler", scaler, minmax_cols)
    ], 
    remainder="passthrough",
    verbose_feature_names_out=False)

    pipeline = Pipeline([
        ("Type", type_transformer),
        ("Encoder", encoder_transformer),
        ("Scaler", scaler_transformer),
        ("Classifier", model)
    ]).set_output(transform="pandas")

    X = train_df.drop(columns=["HiringDecision"])
    y = train_df["HiringDecision"]
    pipeline.fit(X, y)

    joblib.dump(pipeline, f"{home_dir}/{dir}/models/{model_name}_pipeline.joblib")

# lee todos los archivos joblib de la carpeta.
def evaluate_models(**kwargs):
    dir = f"{kwargs.get('ds')}"
    test_df = pd.read_csv(f"{home_dir}/{dir}/splits/test.csv")

    X_test = test_df.drop(columns=["HiringDecision"])
    y_test = test_df["HiringDecision"]
    file_pattern = os.path.join(f"{home_dir}/{dir}/models/", "*.joblib")
    files = glob.glob(file_pattern)

    if not files:
        raise Exception("No joblib files in folder.")

    best = None
    best_score = 0
    best_name = ""
    for file_name in files:
        pipeline = joblib.load(file_name)
        predictions = pipeline.predict(X_test)
        score = accuracy_score(y_true=y_test, y_pred=predictions)
        if score > best_score:
            best_score = score
            best = pipeline
            best_name = file_name.removeprefix(f"{home_dir}/{dir}/models/").removesuffix("_pipeline.joblib")
    
    joblib.dump(pipeline, f"{home_dir}/{dir}/models/best_pipeline.joblib")

    print(f"The best model is {best_name} with an accuracy of {best_score:.2f}!")


