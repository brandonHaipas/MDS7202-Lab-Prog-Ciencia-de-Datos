import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import gradio as gr

set_config(transform_output="pandas")

# misma semilla para slpit y para entrenamiento
seed = 123

def create_folders(**kwargs):
    date = kwargs.get("date")
    os.mkdir(date)
    os.mkdir(f"{date}/raw")
    os.mkdir(f"{date}/splits")
    os.mkdir(f"{date}/models")
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

class TypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, type):
        self.type_to_transform=type
        return self
    
    def fit(self, X, y =None):
        return self
    
    def transform(self, X):
        return X.astype(self.type_to_transform)

def preprocess_and_train(**kwargs):
    dir = f"{kwargs.get('date')}"
    train_df = pd.read_csv(f"{dir}/splits/train.csv")
    test_df = pd.read_csv(f"{dir}/splits/test.csv")

    # basandose en data_1_report, se puede concluir que las unicas variables que necesitan one_hot_encoding son EducationLevel RecruitmentStrategy.
    # el resto de columnas "categoricas" solo necesita un cambio de tipos.
    # para el resto de variables se podría aplicar un scaler, dado que no hay un mayor desbalance se puede usar directamente un minmax scaler.

    encode_cols = ["EducationLevel", "RecruitmentStrategy"]
    astype_cols = ["PreviousCompanies"]
    minmax_cols = ["Age","ExperienceYears","PreviousCompanies", "DistanceFromCompany", "InterviewScore", "SkillScore"]

    encoder = OneHotEncoder()
    scaler = MinMaxScaler(feature_range=(0,1))
    type_transformer = TypeTransformer(int)

    encoding_transformer = ColumnTransformer([
        ("One Hot Encoding", encoder, encode_cols),
        ("Type transformer", type_transformer, astype_cols)
    ],
    remainder="passthrough")

    scaler_transformer = ColumnTransformer([
        ("MinMax Scaler", scaler, minmax_cols)
    ], 
    remainder = "passthrough")
    
    rf = RandomForestClassifier()

    pipeline = Pipeline([
        ("Encoder", encoding_transformer),
        ("Scaler", scaler_transformer),
        ("Random Forest Classifier", rf)
    ])

    X = train_df.drop(columns=["HiringDecision"])
    y = train_df["HiringDecision"]
    pipeline.fit(X,y)

    joblib.dump(pipeline, f"{dir}/models/pipeline.joblib")
    X_test = test_df.drop(columns=["HiringDecision"])
    y_test = test_df["HiringDecision"]
    predictions = pipeline.transform(X_test)

    report = classification_report(y_true=y_test, y_pred=predictions)
    accuracy = report["1"]["accuracy"]
    f1_score = report["1"]["f1-score"]

    print(f"Accuracy for positive class: {accuracy:.2f}")
    print(f"F1-Score for positive class: {f1_score:.2f}")

def predict(file,model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}


def gradio_interface(**kwargs):
    dir = f"{kwargs.get('date')}"
    model_path= f"{dir}/models/pipeline.joblib" #Completar con la ruta del modelo entrenado

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)