# 🍷 Machine Learning Wine Classification: A Comprehensive Analysis using Python, Pandas, NumPy and Scikit-learn

[![English](https://img.shields.io/badge/lang-English-blue.svg)](#english) [![Español](https://img.shields.io/badge/lang-Espa%C3%B1ol-red.svg)](#español)

<a name="english"></a>
## 🇺🇸 English

This repository contains a comprehensive analysis of the Wine dataset using various machine learning classification models. The project compares the performance of different algorithms and identifies the most important features for classifying wine varieties.

### 🔍 Overview

The Wine dataset contains 13 attributes derived from chemical analysis of wines grown in the same region in Italy but from different cultivars. The goal is to classify the wines into one of three classes.

### 🤖 Models Implemented

- 📊 Logistic Regression
- 🔍 K-Nearest Neighbors (KNN)
- 🌳 Decision Tree
- 🧠 Support Vector Machine (SVM)
- 🌲 Random Forest

### 📁 Project Structure

The main notebook (`wine_classification_model_comparison.ipynb`) is organized as follows:

1. **📥 Introduction and Data Loading**
   - Loading the Wine dataset
   - Initial data exploration

2. **📊 Data Exploration and Visualization**
   - Statistical analysis
   - Feature distributions
   - Class distributions
   - Feature relationships

3. **🔧 Data Preparation**
   - Correlation analysis
   - Feature selection
   - Data scaling
   - Train-test splitting

4. **⚙️ Model Training**
   - Implementation of five classification models
   - Individual model evaluation
   - Hyperparameter optimization for KNN

5. **📈 Model Comparison**
   - Performance comparison across models
   - Cross-validation for robust evaluation

6. **🏆 Final Model and Conclusions**
   - Selection of the best performing model
   - Feature importance analysis
   - PCA visualization

### 💡 Key Findings

- 🥇 SVM achieved the highest cross-validation accuracy (98.33%)
- 🥈 Logistic Regression and Random Forest also performed exceptionally well
- 🔑 "Proline", "flavanoids", "alcohol", and "hue" were identified as the most important features
- 🔗 Strong correlation was found between 'total_phenols' and 'flavanoids'

### 📦 Requirements

The project requires the following Python libraries:
- 🐼 pandas
- 🔢 numpy
- 🧪 scikit-learn
- 📉 matplotlib
- 🌊 seaborn

### 🚀 Usage

Clone the repository and run the Jupyter notebook:

```bash
git clone https://github.com/yourusername/machine-learning-wine-classification-python.git
cd machine-learning-wine-classification-python
jupyter notebook wine_classification_model_comparison.ipynb
```

---

<a name="español"></a>
## 🇪🇸 Español

Este repositorio contiene un análisis exhaustivo del conjunto de datos Wine (Vinos) utilizando varios modelos de clasificación de aprendizaje automático. El proyecto compara el rendimiento de diferentes algoritmos e identifica las características más importantes para clasificar variedades de vino.

### 🔍 Descripción General

El conjunto de datos Wine contiene 13 atributos derivados del análisis químico de vinos cultivados en la misma región de Italia pero de diferentes variedades. El objetivo es clasificar los vinos en una de tres clases.

### 🤖 Modelos Implementados

- 📊 Regresión Logística
- 🔍 K-Vecinos Más Cercanos (KNN)
- 🌳 Árbol de Decisión
- 🧠 Máquina de Vectores de Soporte (SVM)
- 🌲 Bosque Aleatorio (Random Forest)

### 📁 Estructura del Proyecto

El cuaderno principal (`wine_classification_model_comparison.ipynb`) está organizado de la siguiente manera:

1. **📥 Introducción y Carga de Datos**
   - Carga del conjunto de datos Wine
   - Exploración inicial de los datos

2. **📊 Exploración y Visualización de Datos**
   - Análisis estadístico
   - Distribuciones de características
   - Distribuciones de clases
   - Relaciones entre características

3. **🔧 Preparación de Datos**
   - Análisis de correlación
   - Selección de características
   - Escalado de datos
   - División en conjuntos de entrenamiento y prueba

4. **⚙️ Entrenamiento de Modelos**
   - Implementación de cinco modelos de clasificación
   - Evaluación individual de modelos
   - Optimización de hiperparámetros para KNN

5. **📈 Comparación de Modelos**
   - Comparación de rendimiento entre modelos
   - Validación cruzada para una evaluación robusta

6. **🏆 Modelo Final y Conclusiones**
   - Selección del mejor modelo
   - Análisis de importancia de características
   - Visualización PCA

### 💡 Hallazgos Clave

- 🥇 SVM logró la mayor precisión de validación cruzada (98.33%)
- 🥈 La Regresión Logística y Random Forest también tuvieron un rendimiento excepcional
- 🔑 "Proline", "flavanoids", "alcohol" y "hue" fueron identificadas como las características más importantes
- 🔗 Se encontró una fuerte correlación entre 'total_phenols' y 'flavanoids'

### 📦 Requisitos

El proyecto requiere las siguientes bibliotecas de Python:
- 🐼 pandas
- 🔢 numpy
- 🧪 scikit-learn
- 📉 matplotlib
- 🌊 seaborn

### 🚀 Uso

Clona el repositorio y ejecuta el cuaderno de Jupyter:

```bash
git clone https://github.com/yourusername/machine-learning-wine-classification-python.git
cd machine-learning-wine-classification-python
jupyter notebook wine_classification_model_comparison.ipynb
```
