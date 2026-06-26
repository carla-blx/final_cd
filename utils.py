# utils.py
"""
Utilidades para la app de predicción de demanda eléctrica.
Dataset: Individual Household Electric Power Consumption (UCI).
Modelos: RNN, LSTM, GRU (Keras) entrenados sobre 20 variables predictoras
agregadas a nivel horario + features de calendario y rezagos.
"""

import pickle
import numpy as np
import pandas as pd
import os
import traceback

# ============================================
# CONFIGURACIÓN GENERAL
# ============================================

FEATURES = [
    'reactive_power', 'voltage', 'intensity', 'sub1', 'sub2', 'sub3',
    'sub_rem', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'dow_sin', 'dow_cos', 'is_weekend', 'lag_1', 'lag_2', 'lag_3',
    'lag_24', 'lag_168', 'roll_mean_24'
]
N_FEATURES = len(FEATURES)  # 20

DEFAULT_TARGET_COL = 'global_active_power'

MODEL_FILES = {
    'RNN':  'modelo_rnn.h5',
    'LSTM': 'modelo_lstm.keras',
    'GRU':  'modelo_gru.keras',
}

METRICS_FILES = {
    'RNN':  'metricas_rnn.pkl',
    'LSTM': 'metricas_lstm.pkl',
    'GRU':  'metricas_gru.pkl',
}

SCALER_FILE = 'scaler.pkl'

# ============================================
# FUNCIÓN PARA OBTENER RUTA BASE
# ============================================
def get_base_path():
    """Obtiene la ruta del directorio donde se encuentra este archivo."""
    return os.path.dirname(os.path.abspath(__file__))

# ============================================
# CARGA DE SCALER
# ============================================
def load_scaler(path: str = SCALER_FILE):
    """Carga el scaler (soporta tanto pickle plano como joblib)."""
    base_path = get_base_path()
    full_path = path if os.path.isabs(path) else os.path.join(base_path, path)
    
    print(f"Cargando scaler desde: {full_path}")
    
    try:
        import joblib
        return joblib.load(full_path)
    except Exception:
        with open(full_path, 'rb') as f:
            return pickle.load(f)

# ============================================
# CARGA DE MODELOS (Keras)
# ============================================
def load_keras_model(path: str):
    """Carga un modelo .h5 o .keras indistintamente."""
    from tensorflow.keras.models import load_model
    
    base_path = get_base_path()
    full_path = path if os.path.isabs(path) else os.path.join(base_path, path)
    
    print(f"Cargando modelo desde: {full_path}")
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"El archivo {full_path} no existe")
    
    return load_model(full_path, compile=False)

def load_all_models(model_files: dict = MODEL_FILES) -> dict:
    """Carga los 3 modelos (RNN, LSTM, GRU) en un diccionario."""
    models = {}
    
    print("=" * 50)
    print("CARGANDO MODELOS")
    print("=" * 50)
    
    for name, path in model_files.items():
        print(f"\n--- Intentando cargar {name} ---")
        try:
            model = load_keras_model(path)
            models[name] = model
            print(f"✅ Modelo {name} cargado correctamente")
        except Exception as e:
            print(f"❌ Error cargando {name}: {str(e)}")
    
    print("=" * 50)
    print(f"RESUMEN: {len(models)} modelos cargados correctamente")
    print("=" * 50)
    
    return models

# ============================================
# CARGA DE MÉTRICAS
# ============================================
def load_metrics(path: str) -> dict:
    """Carga un diccionario de métricas guardado con pickle."""
    base_path = get_base_path()
    full_path = path if os.path.isabs(path) else os.path.join(base_path, path)
    
    with open(full_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, pd.Series):
        data = data.to_dict()
    elif isinstance(data, pd.DataFrame):
        data = data.iloc[0].to_dict()
    return data

def load_all_metrics(metrics_files: dict = METRICS_FILES) -> dict:
    """Carga las métricas de los 3 modelos en un diccionario anidado."""
    metrics = {}
    for name, path in metrics_files.items():
        try:
            metrics[name] = load_metrics(path)
        except Exception as e:
            metrics[name] = {'error': str(e)}
    return metrics

def metrics_to_dataframe(metrics_dict: dict) -> pd.DataFrame:
    """Convierte {modelo: {metrica: valor}} en un DataFrame comparativo."""
    rows = []
    for model_name, m in metrics_dict.items():
        row = {'Modelo': model_name}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)

def get_best_model(metrics_df: pd.DataFrame, metric: str = 'RMSE', minimize: bool = True):
    """Devuelve el nombre del modelo con el mejor valor para una métrica dada."""
    if metric not in metrics_df.columns:
        return None
    col = pd.to_numeric(metrics_df[metric], errors='coerce')
    idx = col.idxmin() if minimize else col.idxmax()
    if pd.isna(idx):
        return None
    return metrics_df.loc[idx, 'Modelo']

# ============================================
# VALIDACIÓN DE DATOS DE ENTRADA
# ============================================
def validate_dataframe(df: pd.DataFrame, features: list = FEATURES) -> list:
    """Devuelve la lista de columnas predictoras que faltan en el df."""
    return [c for c in features if c not in df.columns]

# ============================================
# FORMA DE ENTRADA ESPERADA POR EL MODELO
# ============================================
def get_model_input_shape(model):
    """Devuelve (timesteps, n_features) leyendo el input_shape del modelo."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    timesteps = shape[1]
    n_feat = shape[2] if len(shape) > 2 else None
    return timesteps, n_feat

# ============================================
# ESCALADO Y CONSTRUCCIÓN DE SECUENCIAS
# ============================================
def scale_features(df: pd.DataFrame, scaler, target_col: str, features: list = FEATURES,
                   scaler_includes_target: bool = True) -> np.ndarray:
    """Escala los datos con el scaler entrenado.

    Si scaler_includes_target=True, el scaler fue entrenado con [target] + features
    (21 columnas, target primero). Por lo tanto hay que pasarle a `transform`
    un array con esas mismas 21 columnas y EN ESE ORDEN -- no se le puede pasar
    solo las 20 predictoras, porque el scaler exige n_features_in_ == 21.

    Además, el modelo (RNN/LSTM/GRU) espera 21 variables por paso de tiempo
    (el target histórico es una entrada más de la secuencia), así que se
    devuelven las 21 columnas ya escaladas, no solo las 20 predictoras.
    """
    if scaler_includes_target:
        all_cols = [target_col] + features
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Faltan columnas para escalar (target + predictoras): {missing}"
            )

        # Array con las 21 columnas, en el MISMO orden usado al entrenar el scaler
        X_full = df[all_cols].values  # (n_samples, 21)

        # Una sola llamada a transform con las 21 columnas -> sin error de shape
        X_scaled_full = scaler.transform(X_full)  # (n_samples, 21)

        return X_scaled_full
    else:
        # Escalar solo las predictoras (scaler entrenado sin el target)
        return scaler.transform(df[features].values)

def build_sequence(X_scaled: np.ndarray, timesteps: int, scaler_includes_target: bool = True) -> np.ndarray:
    """Toma las últimas `timesteps` filas y construye el array para el modelo."""
    if len(X_scaled) < timesteps:
        raise ValueError(
            f"Se necesitan al menos {timesteps} filas históricas para predecir "
            f"(hay {len(X_scaled)}). Sube un histórico más largo."
        )
    
    # Tomar las últimas timesteps filas
    data = X_scaled[-timesteps:]
    
    # Reshape para el modelo: (1, timesteps, n_features)
    return data.reshape(1, timesteps, data.shape[1])

# ============================================
# DES-ESCALADO DE LA PREDICCIÓN (TARGET)
# ============================================

def inverse_transform_target(pred_scaled_value: float, scaler, n_features_scaler: int,
                              target_index: int = 0, scaler_includes_target: bool = True) -> float:
    """
    Des-escala el valor predicho del target.
    """
    if not scaler_includes_target:
        return float(pred_scaled_value)

    # Crear un vector dummy con el tamaño correcto
    dummy = np.zeros((1, n_features_scaler))
    # Colocar el valor predicho en la posición del target
    dummy[0, target_index] = pred_scaled_value
    # Inversa transform
    inv = scaler.inverse_transform(dummy)
    return float(inv[0, target_index])

# === AÑADE ESTA FUNCIÓN ADICIONAL ===
def inverse_transform_sequence(scaled_sequence: np.ndarray, scaler, target_index: int = 0) -> np.ndarray:
    """
    Des-escala una secuencia completa para visualización
    """
    inv_sequence = scaler.inverse_transform(scaled_sequence)
    return inv_sequence[:, target_index]

# ============================================
# PREDICCIÓN
# ============================================
def predict_single(model, scaler, df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL,
                    features: list = FEATURES, timesteps: int = None,
                    scaler_includes_target: bool = True):
    """Ejecuta una predicción puntual (siguiente paso) con un modelo dado."""
    if timesteps is None:
        timesteps, _ = get_model_input_shape(model)

    # Escalar los datos incluyendo o no el target (21 o 20 columnas según el caso)
    X_scaled = scale_features(df, scaler, target_col, features, scaler_includes_target)
    
    # Construir la secuencia
    X = build_sequence(X_scaled, timesteps, scaler_includes_target)

    pred_scaled = model.predict(X, verbose=0)
    pred_scaled_value = float(np.ravel(pred_scaled)[0])
    return pred_scaled_value

def predict_with_model(model, scaler, df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL,
                        features: list = FEATURES, timesteps: int = None,
                        scaler_includes_target: bool = True, target_index: int = 0):
    """Predice el siguiente valor de consumo y lo devuelve en escala real
    junto con el valor en escala normalizada (por transparencia/debug)."""
    pred_scaled_value = predict_single(model, scaler, df, target_col, features, timesteps, scaler_includes_target)

    n_feat_scaler = getattr(
        scaler, 'n_features_in_',
        len(features) + (1 if scaler_includes_target else 0)
    )
    pred_real = inverse_transform_target(
        pred_scaled_value, scaler, n_feat_scaler,
        target_index=target_index,
        scaler_includes_target=scaler_includes_target
    )
    return pred_real, pred_scaled_value

def predict_all_models(models: dict, scaler, df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL,
                        features: list = FEATURES, **kwargs) -> dict:
    """Ejecuta predict_with_model para cada modelo cargado (RNN, LSTM, GRU)."""
    results = {}
    for name, model in models.items():
        if name == '_errors':
            continue
        try:
            real, scaled = predict_with_model(model, scaler, df, target_col, features, **kwargs)
            results[name] = {'prediccion': real, 'prediccion_escalada': scaled}
        except Exception as e:
            results[name] = {'error': str(e)}
    return results

# ============================================
# FORMATO
# ============================================
def format_kw(value: float) -> str:
    """Formatea un valor de potencia en kW con 3 decimales."""
    return f"{value:,.3f} kW"