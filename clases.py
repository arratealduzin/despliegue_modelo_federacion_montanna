# ─────────────────────────────────────────────
# 1. TRANSFORMADORES PERSONALIZADOS
# ─────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin

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