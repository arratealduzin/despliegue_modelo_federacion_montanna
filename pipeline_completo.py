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

class DropLeakageColumns(BaseEstimator, TransformerMixin):
    """Elimina columnas con data leakage y columnas innecesarias."""

    DROP_COLS = [
        "Actividad", "Lugar", "Provincia", "TipoAccidente",
        "DescripcionGrado", "TamañoGrupo", "NResponsables",
        "Entrenamiento", "ActividadPersonal", "ActividadOrganizada",
        "Festivo", "año_accidente", "hospitalizacion", "helicoptero",
        "discapacidad", "numeroaccidentesaño",
        "nsocio", "numeroclub",
    ]

    def fit(self, X, y=None):
        # Solo eliminamos las que existan en el dataframe
        self.cols_to_drop_ = [c for c in self.DROP_COLS if c in X.columns]
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_, errors="ignore")


class FixCategoriaEncoding(BaseEstimator, TransformerMixin):
    """Corrige el encoding latin1/utf-8 de la columna 'categoria'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "categoria" in X.columns:
            X["categoria"] = (
                X["categoria"]
                .str.encode("latin1")
                .str.decode("utf-8")
            )
        return X


class CreateGrupoPersona(BaseEstimator, TransformerMixin):
    """
    - Crea la columna 'grupo_persona' a partir de 'edad' y 'categoria'.
    - Mapea 'modalidad' a descripciones legibles.
    - Elimina 'categoria' (ya no necesaria).
    """

    MAPEO_MODALIDAD = {
        "A": "Senderismo", "A6": "Senderismo España",
        "OT": "actividades otoño", "OTPromo": "actividades otoño",
        "AU": "Montaña C.A.Madrid",
        "B": "Montaña", "B6": "Montaña", "B7": "Montaña",
        "B Comp.": "Montaña competición",
        "C": "Montaña Europa y Marruecos",
        "ROC": "Rocodromo",
        "D": "Montaña < 7000m mundo",
        "D Comp.": "Montaña < 7000m mundo competición",
        "E": "Montaña > 7000m",
        "E Comp.": "Montaña > 7000m competición",
    }

    @staticmethod
    def _grupo_persona(row):
        edad = row["edad"]
        cat = row.get("categoria", "")
        sexo = "Mujer" if "Mujer" in str(cat) else "Hombre"
        if edad < 14:
            return f"Niño/a {sexo}"
        elif edad < 18:
            return f"Juvenil {sexo}"
        return sexo

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "categoria" in X.columns:
            X["grupo_persona"] = X.apply(self._grupo_persona, axis=1)
            X = X.drop(columns=["categoria"])

        if "modalidad" in X.columns:
            X["modalidad"] = X["modalidad"].map(self.MAPEO_MODALIDAD)

        return X


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
    df = pd.read_csv("dataset_definitivo.csv")

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
