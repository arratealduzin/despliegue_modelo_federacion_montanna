import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# ─────────────────────────────────────────────
# 1. TRANSFORMADORES PERSONALIZADOS
# ─────────────────────────────────────────────
from clases import DropLeakageColumns, FixCategoriaEncoding, CreateGrupoPersona


# ─────────────────────────────────────────────
# 2. PREPROCESAMIENTO SKLEARN
# ─────────────────────────────────────────────

CAT_COLS = ["modalidad", "grupo_persona"]
NUM_COLS = ["edad", "accidenteshistoricos", "temporada_federado"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_COLS),
        ("cat", categorical_transformer, CAT_COLS),
    ],
    remainder="drop",
)


# ─────────────────────────────────────────────
# 3. PIPELINE COMPLETO
# ─────────────────────────────────────────────

pipeline_completo = Pipeline([
    ("drop_leakage",     DropLeakageColumns()),
    ("fix_encoding",     FixCategoriaEncoding()),
    ("feature_eng",      CreateGrupoPersona()),
    ("preprocess",       preprocessor),
    ("model",            LogisticRegression(
                             max_iter=5000,
                             class_weight="balanced",
                             C=1,
                             solver="lbfgs",
                         )),
])


# ─────────────────────────────────────────────
# 4. USO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Carga el CSV en crudo (con todas las columnas originales)
    df = pd.read_csv("./data/dataset_definitivo.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    from sklearn.model_selection import GroupShuffleSplit
    groups = df["nsocio"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Entrena el pipeline completo de una sola vez
    pipeline_completo.fit(X_train, y_train)

    # Evalúa
    from sklearn.metrics import roc_auc_score
    y_prob = pipeline_completo.predict_proba(X_test)[:, 1]
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    # Guarda
    joblib.dump(pipeline_completo, "pipeline_completo.pkl")
    print("Pipeline guardado en pipeline_completo.pkl")

    # Para hacer predicciones en producción con datos nuevos:
    # pipeline = joblib.load("pipeline_completo.pkl")
    # predicciones = pipeline.predict_proba(df_nuevos)[:, 1]
