# ⚡ Predicción de Demanda Eléctrica

Aplicación web interactiva para la predicción de consumo eléctrico utilizando modelos de deep learning (RNN, LSTM y GRU) entrenados con el dataset **Individual Household Electric Power Consumption** de UCI.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Dataset](#-dataset)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Uso](#-uso)
- [Modelos](#-modelos)
- [Métricas de Desempeño](#-métricas-de-desempeño)
- [Visualización](#-visualización)
- [Despliegue](#-despliegue)
- [Contribución](#-contribución)
- [Licencia](#-licencia)
- [Contacto](#-contacto)

## 🚀 Características

- **Tres Modelos de Deep Learning**: RNN, LSTM y GRU para comparación de rendimiento
- **Interfaz Intuitiva**: Desarrollada con Streamlit para fácil interacción
- **Visualización en Tiempo Real**: Gráficos interactivos con Plotly
- **Comparación de Modelos**: Visualización lado a lado de predicciones
- **Métricas de Rendimiento**: RMSE, MAE, MSE y R² para cada modelo
- **Carga de Datos Personalizada**: Soporte para archivos CSV con datos históricos
- **Escalado Automático**: Pipeline completo de preprocesamiento de datos

## 📊 Dataset

El dataset utilizado es el **Individual Household Electric Power Consumption** del repositorio UCI Machine Learning.

### Características del Dataset:
- **Fuente**: UCI Machine Learning Repository
- **Periodo**: Diciembre 2006 - Noviembre 2010
- **Granularidad**: Datos por minuto (agregados a nivel horario)
- **Variables**: 21 variables (1 objetivo + 20 predictoras)

### Variables Predictoras (20):
1. `reactive_power` - Potencia reactiva
2. `voltage` - Voltaje
3. `intensity` - Intensidad de corriente
4. `sub1` - Sub-medición 1 (cocina)
5. `sub2` - Sub-medición 2 (lavandería)
6. `sub3` - Sub-medición 3 (calentador/aire acondicionado)
7. `sub_rem` - Sub-medición restante
8. `hour_sin` - Seno de la hora (codificación circular)
9. `hour_cos` - Coseno de la hora (codificación circular)
10. `month_sin` - Seno del mes (codificación circular)
11. `month_cos` - Coseno del mes (codificación circular)
12. `dow_sin` - Seno del día de la semana (codificación circular)
13. `dow_cos` - Coseno del día de la semana (codificación circular)
14. `is_weekend` - Indicador de fin de semana (0/1)
15. `lag_1` - Rezago de 1 hora
16. `lag_2` - Rezago de 2 horas
17. `lag_3` - Rezago de 3 horas
18. `lag_24` - Rezago de 24 horas
19. `lag_168` - Rezago de 168 horas (7 días)
20. `roll_mean_24` - Media móvil de 24 horas

### Variable Objetivo:
- `global_active_power` - Consumo activo global (kW)

## 🏗️ Arquitectura del Sistema
