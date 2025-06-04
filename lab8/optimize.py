# Optimize.py
import os
import mlflow
import pickle
import json

import optuna
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

import pandas as pd
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Seed
random_seed = 9999

# Definiciones
water_df = pd.read_csv('water_potability.csv')

X = water_df.drop(columns=['Potability'])
y = water_df['Potability']

X_train, X_test_val, y_train, y_test_val = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=random_seed,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test_val,
    y_test_val,
    test_size=1/3,
    random_state=random_seed,
)

# Experimento
n_experiments = len(mlflow.search_experiments())
experiment = mlflow.create_experiment(f"XGBoost optimization with optuna n°{n_experiments}")

# Crear carpetas de resultados
os.mkdir(f'models/exp_{experiment}')
os.mkdir(f'plots/exp_{experiment}')

def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    print(runs)
    best_model_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    with open(f'models/exp_{experiment_id}/model.pkl','wb') as f:
        pickle.dump(best_model, f)

    return best_model


def objective(trial):
    # Hiperparametros a optimizar
    xgb_params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "max_leaves": trial.suggest_int("max_leaves", 0, 100),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
    }

    # Definición de pipeline
    transformer = ColumnTransformer(
        [
            (
                "Scale",
                StandardScaler(),
                ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"],
            ),
        ],
    )

    pipeline = Pipeline(
        steps=[
            ("Preprocessing", transformer),
            ("Classifier", XGBClassifier(seed=random_seed, **xgb_params))
        ]
    )

    with mlflow.start_run(run_name=f"XGBoost with {xgb_params}", experiment_id=experiment):
        # Evaluación
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_val)
        f1 = f1_score(y_val, pred)

        mlflow.log_metric("valid_f1", f1)

    return f1

def optimize_model():

    # Optimización
    mlflow.autolog()
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_seed))
    study.optimize(objective, timeout=300)

    # Obtención del mejor modelo
    best_model_pipeline = get_best_model(experiment)
    best_model = best_model_pipeline.named_steps["Classifier"]
    preprocessor = best_model_pipeline.named_steps["Preprocessing"]

    # Gráficos de Optuna
    plot_optimization_history(study).get_figure().savefig(f"plots/exp_{experiment}/optimization_history.png")
    plot_param_importances(study).get_figure().savefig(f"plots/exp_{experiment}/param_importances.png")

    # Respaldo de configuraciones de modelo
    with open(f"models/exp_{experiment}/best_config.json", "w") as f:
        json.dump(best_model.get_xgb_params(), f, indent=4)

    # Respaldo de importancias
    importances = best_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    # Plotear feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importances)), importances)
    plt.yticks(range(len(importances)), feature_names)
    plt.ylabel("Features")
    plt.xlabel("Importance")
    plt.title(f"Best XGBClassifier Feature Importances in experiment exp_{experiment}")
    plt.tight_layout()
    plt.savefig(f"plots/exp_{experiment}/feature_importances.png")
    plt.close()

if __name__ == "__main__":
    optimize_model()
