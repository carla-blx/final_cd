# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import traceback

# Asegurar que el directorio actual está en el path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils import (
    FEATURES,
    DEFAULT_TARGET_COL,
    MODEL_FILES,
    METRICS_FILES,
    load_scaler,
    load_all_models,
    load_all_metrics,
    validate_dataframe,
    get_model_input_shape,
    predict_with_model,
    predict_all_models,
    metrics_to_dataframe,
    get_best_model,
    format_kw,
)

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Predicción de Demanda Eléctrica",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado (tema eléctrico)
st.markdown("""
<style>
:root {
    --primary: #0d47a1;
    --secondary: #1976d2;
    --accent: #ffc107;
    --bg: #eef3fb;
    --text: #1b1b1b;
}
 
.stApp {
    background-color: var(--bg);
    color: var(--text);
}
 
h1.main-header {
    color: var(--primary);
    text-align: center;
    font-weight: 800;
    margin-bottom: 10px;
    font-size: 2.4rem;
}
 
.prediction-box {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    padding: 22px;
    border-radius: 12px;
    color: white;
    font-size: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}
 
.metric-card {
    background: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}
 
.stButton>button {
    background-color: var(--primary);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s;
}
 
.stButton>button:hover {
    background-color: var(--secondary);
    color: white;
    transform: translateY(-2px);
}
 
.stAlert {
    border-radius: 10px;
}
 
.best-badge {
    background-color: var(--accent);
    color: #1b1b1b;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# TÍTULO
# ============================================
st.markdown("<h1 class='main-header'>⚡ Predicción de Demanda Eléctrica</h1>",
            unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; margin-bottom: 25px; font-size: 1.05rem;'>
    Dataset <b>Individual Household Electric Power Consumption (UCI)</b> · Modelos
    recurrentes <b>RNN, LSTM y GRU</b> entrenados sobre datos horarios para predecir
    el siguiente valor de consumo (<code>global_active_power</code>).
    </div>
""", unsafe_allow_html=True)

# ============================================
# CARGA DE RECURSOS (cacheados)
# ============================================
@st.cache_resource
def get_scaler():
    return load_scaler()

@st.cache_resource
def get_models():
    models = load_all_models()
    return models

@st.cache_data
def get_metrics():
    return load_all_metrics()

# Mostrar información de depuración
with st.expander("🔧 Información de depuración", expanded=True):  # Cambiado a expanded=True
    st.write(f"Directorio actual: {os.getcwd()}")
    st.write(f"Directorio del script: {current_dir}")
    st.write(f"Python path: {sys.path[:3]}...")
    
    st.write("\n📁 Archivos en el directorio:")
    files_found = []
    for file in os.listdir(current_dir):
        if file.endswith(('.h5', '.keras', '.pkl')):
            files_found.append(file)
            st.write(f"  - {file}")
    
    if not files_found:
        st.error("⚠️ No se encontraron archivos de modelo en el directorio")

# Cargar scaler
try:
    with st.spinner("🔄 Cargando scaler..."):
        scaler = get_scaler()
    st.success("✅ Scaler cargado correctamente")
except Exception as e:
    st.error(f"❌ Error al cargar el scaler: {str(e)}")
    st.code(traceback.format_exc())
    st.stop()

# Cargar modelos
try:
    with st.spinner("🔄 Cargando modelos..."):
        models = get_models()
    
    # Verificar qué modelos se cargaron
    if models:
        model_names = list(models.keys())
        st.success(f"✅ Modelos cargados correctamente: {', '.join(model_names)}")
        st.write(f"Total de modelos cargados: {len(models)}")
    else:
        st.error("❌ No se pudo cargar ningún modelo")
        st.info("""
        Posibles soluciones:
        1. Verifica que TensorFlow esté instalado: `pip install tensorflow`
        2. Asegúrate de que los archivos .h5 y .keras no estén corruptos
        3. Revisa la consola para ver errores detallados
        """)
        st.stop()
        
except Exception as e:
    st.error(f"❌ Error al cargar los modelos: {str(e)}")
    st.code(traceback.format_exc())
    st.info("Verifica que los archivos de modelo existan y sean válidos.")
    st.stop()

# Cargar métricas
try:
    with st.spinner("🔄 Cargando métricas..."):
        metrics_dict = get_metrics()
    metrics_df = metrics_to_dataframe(metrics_dict)
    
    if not metrics_df.empty and 'error' not in metrics_df.columns:
        st.success(f"✅ Métricas cargadas correctamente para: {', '.join(metrics_dict.keys())}")
    else:
        st.warning("⚠️ No se encontraron métricas válidas para los modelos")
        
except Exception as e:
    metrics_dict, metrics_df = {}, pd.DataFrame()
    st.warning(f"⚠️ No se pudieron cargar las métricas: {str(e)}")

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## ℹ️ Información")
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown("""
    - **Fuente:** UCI - Individual Household Electric Power Consumption
    - **Granularidad:** Horaria (agregada desde minutos)
    - **Predictoras:** 20 variables (potencia, voltaje, sub-mediciones,
      variables de calendario y rezagos)
    """)

    st.markdown("---")
    st.markdown("### 🧠 Modelos disponibles")
    if models:
        for name in MODEL_FILES:
            status = "✅" if name in models else "❌"
            st.markdown(f"- {status} **{name}**")
    else:
        st.markdown("⚠️ No hay modelos cargados")

    st.markdown("---")
    st.markdown("### ⚙️ Configuración de datos")
    target_col = st.text_input("Nombre de la columna objetivo", value=DEFAULT_TARGET_COL)

    st.markdown("**Configuración del scaler** (ajusta si tu pipeline difiere)")
    scaler_includes_target = st.checkbox(
        "El scaler fue ajustado incluyendo el target junto a las predictoras",
        value=True,
        help="Patrón típico en pipelines LSTM: se escala [target + 20 predictoras] "
             "juntos. Desactívalo si tu target no pasó por este scaler."
    )
    target_index = st.number_input(
        "Posición (índice) del target dentro del scaler", min_value=0, value=0, step=1
    )

    st.markdown("---")
    st.markdown("### 📖 Instrucciones")
    st.markdown("""
    1. Sube un CSV horario con las 20 columnas predictoras (ya preprocesadas
       con tu pipeline de PySpark).
    2. Ve a la pestaña **Predicción** o **Comparar modelos**.
    3. Revisa las métricas históricas en la pestaña **Métricas**.
    """)

# ============================================
# CARGA DE DATOS DE ENTRADA
# ============================================
st.markdown("### 📤 Carga de datos históricos")
uploaded_file = st.file_uploader(
    "Sube un CSV con el histórico horario (debe incluir las 20 columnas predictoras)",
    type=["csv"],
    help="El archivo debe tener al menos tantas filas como el 'timesteps' que "
         "espera el modelo (ventana temporal usada en el entrenamiento)."
)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Intentar detectar y ordenar por una columna de fecha si existe
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'fecha' in c.lower()
                     or 'time' in c.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df = df.sort_values(date_cols[0]).reset_index(drop=True)

        missing = validate_dataframe(df, FEATURES)
        if missing:
            st.error(f"❌ Faltan columnas predictoras en el CSV: {missing}")
            df = None
        else:
            st.success(f"✅ Datos cargados: {df.shape[0]} filas × {df.shape[1]} columnas")
            with st.expander("👀 Vista previa de los datos"):
                st.dataframe(df.tail(10), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error al leer el CSV: {str(e)}")
        df = None
else:
    st.info("👈 Sube un archivo CSV para habilitar predicciones y comparaciones.")

st.markdown("---")

# ============================================
# TABS PRINCIPALES
# ============================================
tab1, tab2, tab3 = st.tabs(["🔮 Predicción", "⚖️ Comparar Modelos", "📊 Métricas"])

# --------------------------------------------
# TAB 1: PREDICCIÓN
# --------------------------------------------
with tab1:
    st.markdown("### 🔮 Ejecutar predicción con un modelo")

    if df is None:
        st.info("Sube primero un CSV válido en la sección de arriba.")
    elif not models:
        st.error("No hay modelos cargados disponibles.")
    else:
        model_name = st.radio("Selecciona el modelo", list(models.keys()), horizontal=True)
        model = models[model_name]
        timesteps, n_feat_model = get_model_input_shape(model)
        st.caption(f"Este modelo espera una secuencia de **{timesteps} pasos** "
                   f"× **{n_feat_model} variables**.")

        if st.button("⚡ Ejecutar predicción", key="btn_pred_single"):
            if len(df) < timesteps:
                st.error(f"El CSV tiene {len(df)} filas; se necesitan al menos "
                         f"{timesteps} para construir la secuencia de entrada.")
            else:
                with st.spinner("Calculando predicción..."):
                    try:
                        pred_real, pred_scaled = predict_with_model(
                            model, scaler, df, target_col, FEATURES,
                            timesteps=timesteps,
                            scaler_includes_target=scaler_includes_target,
                            target_index=int(target_index),
                        )
                        st.markdown(f"""
                        <div class='prediction-box'>
                        <span style='font-size: 22px; font-weight: bold;'>
                        Modelo: {model_name}</span><br><br>
                        <b>🔋 Consumo predicho (siguiente hora):</b> {format_kw(pred_real)}<br>
                        <b>📐 Valor en escala normalizada:</b> {pred_scaled:.4f}
                        </div>
                        """, unsafe_allow_html=True)

                        # Gráfico: histórico del target + punto predicho
                        if target_col in df.columns:
                            hist = df[target_col].values[-min(168, len(df)):]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=hist, mode='lines', name='Histórico',
                                line=dict(color='#1976d2')
                            ))
                            fig.add_trace(go.Scatter(
                                x=[len(hist)], y=[pred_real], mode='markers',
                                name=f'Predicción {model_name}',
                                marker=dict(color='#ffc107', size=14, symbol='star')
                            ))
                            fig.update_layout(
                                title="Histórico reciente vs. predicción",
                                xaxis_title="Horas (relativo)",
                                yaxis_title=target_col,
                                template="plotly_white",
                                height=420,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"La columna objetivo '{target_col}' no está en "
                                    "el CSV, así que no se muestra el histórico graficado.")
                    except Exception as e:
                        st.error(f"❌ Error al predecir: {str(e)}")
                        st.code(traceback.format_exc())

# --------------------------------------------
# TAB 2: COMPARAR MODELOS
# --------------------------------------------
with tab2:
    st.markdown("### ⚖️ Comparar predicciones de los 3 modelos")

    if df is None:
        st.info("Sube primero un CSV válido en la sección de arriba.")
    elif not models:
        st.error("No hay modelos cargados disponibles.")
    else:
        if st.button("⚡ Ejecutar comparación", key="btn_pred_compare"):
            min_timesteps = max(
                get_model_input_shape(m)[0] for m in models.values()
            )
            if len(df) < min_timesteps:
                st.error(f"El CSV tiene {len(df)} filas; al menos uno de los modelos "
                         f"necesita {min_timesteps} filas históricas.")
            else:
                with st.spinner("Calculando predicciones de los 3 modelos..."):
                    results = predict_all_models(
                        models, scaler, df, target_col, FEATURES,
                        scaler_includes_target=scaler_includes_target,
                        target_index=int(target_index),
                    )

                cols = st.columns(len(results))
                for col, (name, res) in zip(cols, results.items()):
                    with col:
                        if 'error' in res:
                            st.markdown(f"""
                            <div class='metric-card'>
                            <b>{name}</b><br>
                            <span style='color:red;'>❌ {res['error']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='metric-card'>
                            <b>{name}</b><br>
                            <span style='font-size: 22px; color:#0d47a1; font-weight:700;'>
                            {format_kw(res['prediccion'])}</span><br>
                            <span style='font-size: 0.85rem; color: #666;'>
                            escala normalizada: {res['prediccion_escalada']:.4f}</span>
                            </div>
                            """, unsafe_allow_html=True)

                # Gráfico de barras comparando las predicciones
                valid_results = {k: v['prediccion'] for k, v in results.items() if 'prediccion' in v}
                if valid_results:
                    comp_df = pd.DataFrame({
                        'Modelo': list(valid_results.keys()),
                        'Predicción': list(valid_results.values())
                    })
                    fig = px.bar(
                        comp_df, x='Modelo', y='Predicción', color='Modelo',
                        text_auto='.3f', title="Comparación de predicciones por modelo",
                        color_discrete_sequence=['#0d47a1', '#1976d2', '#ffc107']
                    )
                    fig.update_layout(template="plotly_white", height=420)
                    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
# TAB 3: MÉTRICAS
# --------------------------------------------
with tab3:
    st.markdown("### 📊 Métricas de desempeño (conjunto de prueba)")

    if metrics_df.empty:
        st.info("No hay métricas cargadas. Verifica los archivos "
                f"{list(METRICS_FILES.values())}.")
    else:
        st.dataframe(metrics_df.set_index('Modelo'), use_container_width=True)

        numeric_metrics = [c for c in metrics_df.columns if c != 'Modelo'
                           and pd.api.types.is_numeric_dtype(metrics_df[c])]

        if numeric_metrics:
            metric_choice = st.selectbox(
                "Métrica para resaltar el mejor modelo",
                numeric_metrics,
                index=0
            )
            minimize = st.checkbox(
                "Menor valor = mejor (RMSE/MAE/MSE). Desmarca para métricas tipo R²",
                value=not metric_choice.upper().startswith('R2')
            )
            best = get_best_model(metrics_df, metric_choice, minimize=minimize)
            if best:
                st.markdown(
                    f"🏆 **Mejor modelo según {metric_choice}:** "
                    f"<span class='best-badge'>{best}</span>",
                    unsafe_allow_html=True
                )

            st.markdown("#### Comparación visual de métricas")
            n_cols = min(len(numeric_metrics), 3)
            grid_cols = st.columns(n_cols)
            for i, metric in enumerate(numeric_metrics):
                with grid_cols[i % n_cols]:
                    fig = px.bar(
                        metrics_df, x='Modelo', y=metric, color='Modelo',
                        text_auto='.4f', title=metric,
                        color_discrete_sequence=['#0d47a1', '#1976d2', '#ffc107']
                    )
                    fig.update_layout(template="plotly_white", height=350,
                                       showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se encontraron columnas numéricas en las métricas cargadas.")

# ============================================
# PIE DE PÁGINA
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>Desarrollado con:</b> Streamlit, TensorFlow/Keras, PySpark, Plotly<br>
    <b>Dataset:</b> UCI Individual Household Electric Power Consumption | <b>Versión:</b> 1.0.0
</div>
""", unsafe_allow_html=True)