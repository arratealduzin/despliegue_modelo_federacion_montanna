from flask import Flask, jsonify, request
import joblib
import os
import pandas as pd
import pickle
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from clases import DropLeakageColumns, FixCategoriaEncoding, CreateGrupoPersona

os.chdir(os.path.dirname(os.path.abspath(__file__))) 
#garantiza que el script siempre encuentre el pipeline_completo.pkl independientemente desde dónde se ejecute.

app = Flask(__name__)

# Carga el modelo

model = joblib.load("pipeline_completo.pkl")

# Texto de bienvenida

@app.route("/", methods=["GET"])
def hello():
    return """
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px;">
        <h1>API — Predicción de Accidentes en Federados de Montaña</h1>
        <p>Bienvenid@ a la API del modelo para estimar la probabilidad de accidente en federados de montaña.</p>
 
        <hr>
 
        <h2>Endpoints disponibles</h2>
 
        <h3>GET /api/v1/predict</h3>
        <p>Devuelve la predicción de accidente para un federado dado.</p>
        <b>Parámetros:</b>
        <ul>
            <li><code>categoria</code> — Categoría del federado (ej: <i>Hombre</i>, <i>Mujer</i>)</li>
            <li><code>edad</code> — Edad del federado (número entero)</li>
            <li><code>modalidad</code> — Código de modalidad (ej: <i>B</i>, <i>A</i>, <i>C</i>)</li>
            <li><code>accidenteshistoricos</code> — Número de accidentes previos</li>
            <li><code>temporada_federado</code> — Número de temporadas federado</li>
        </ul>
        <b>Ejemplo:</b>
        <pre style="background:#f4f4f4; padding:10px; border-radius:6px;">
GET /api/v1/predict?categoria=Hombre&edad=35&modalidad=B&accidenteshistoricos=0&temporada_federado=3</pre>
        <b>Respuesta:</b>
        <pre style="background:#f4f4f4; padding:10px; border-radius:6px;">
{
  "prediction": 0,
  "probability_accident": 0.0423
}</pre>
 
        <hr>
 
        <h3>GET /api/v1/retrain</h3>
        <p>Reentrena el modelo con el dataset disponible en el servidor (<code>dataset_definitivo.csv</code>) y devuelve el nuevo AUC-ROC.</p>
        <b>Ejemplo:</b>
        <pre style="background:#f4f4f4; padding:10px; border-radius:6px;">
GET /api/v1/retrain</pre>
        <b>Respuesta:</b>
        <pre style="background:#f4f4f4; padding:10px; border-radius:6px;">
{
  "status": "Modelo reentrenado correctamente",
  "auc_roc": 0.6846
}</pre>
    </body>
    </html>
    """    

# Primer endpoint
@app.route("/api/v1/predict", methods=["GET"])
def predict():
    categoria = request.args.get('categoria', None, type=str)
    edad = request.args.get('edad', None, type=int)
    modalidad = request.args.get('modalidad', None, type=str)
    accidenteshistoricos  = request.args.get("accidenteshistoricos",  None, type=int)
    temporada_federado    = request.args.get("temporada_federado",    None, type=int)

    # Detecta los que faltan (sin usar np.isnan sobre strings)
    params = {
        "categoria":            categoria,
        "edad":                 edad,
        "modalidad":            modalidad,
        "accidenteshistoricos": accidenteshistoricos,
        "temporada_federado":   temporada_federado,
    }
    missing = [k for k, v in params.items() if v is None]

    input_data = pd.DataFrame([params])
    # Predice probabilidad de accidente (clase 1)
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(model.predict(input_data)[0])

    response = {'predictions': pred,
                'probabilidad_accidente': round(float(prob), 4),
                }
    if missing:
        response['warning'] = f"Valores imputados automáticamente para: {', '.join(missing)}"

    return jsonify(response)

# Segundo endpoint reentrenamiento (no creo que podamos usarlo, no hay más datos)
@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    global model
    if not os.path.exists("./data/dataset_definitivo.csv"):
        return jsonify({"error": "dataset_definitivo.csv no encontrado. Nada se ha hecho."}), 404
    data = pd.read_csv('.data/dataset_definitivo.csv')
    data.columns = [col.lower() for col in data.columns]
    X = data.drop(columns=["target"])
    y = data["target"]

    # Split manteniendo socios separados entre train y test    
    groups = data["nsocio"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

     # Reentrena el pipeline completo
    model.fit(X_train, y_train)
 
    # Evalúa solo sobre el conjunto de test
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
 
    # Guarda el modelo actualizado
    joblib.dump(model, "pipeline_completo.pkl")
 
    return jsonify({
        "status":  "Modelo reentrenado correctamente",
        "auc_roc": round(auc, 4),
    })
 
 
if __name__ == "__main__":
    app.run(debug=True)