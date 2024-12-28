import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import shap
import xgboost as xgb


# 1. Carga de datos
ruta_archivo = "C:\\Users\\Antonio\\Downloads\\parkinsons_disease_data.csv"
data = pd.read_csv(ruta_archivo)

# 2. Identificar y eliminar valores nulos
print("Valores nulos por columna:")
print(data.isnull().sum())

# Eliminar valores nulos
data = data.dropna()

# 3. Codificar columnas categóricas (si existen)
categorical_columns = data.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    print("\nColumnas categóricas detectadas:", categorical_columns)
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# 4. Selección factores (X) y objetivo (y)
X = data.drop(columns=['Diagnosis', 'PatientID'])
y = data['Diagnosis']  # Variable objetivo

# 5. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nDatos divididos:")
print(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")

# 6. Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 7. Realizar predicciones
y_pred = rf_model.predict(X_test)

# 8. Evaluación del modelo
print("Informe de clasificación para Random Forest:")
print(classification_report(y_test, y_pred))
print(f"Precisión: {accuracy_score(y_test, y_pred)}")

# Importancia de las características
importances = rf_model.feature_importances_
feature_names = X_train.columns
sorted_indices = importances.argsort()[::-1]
feature_names_sorted = feature_names[sorted_indices]
importances_sorted = importances[sorted_indices]

# Visualizar la importancia de características con gráfico
plt.figure(figsize=(12, 10))
plt.barh(feature_names_sorted, importances_sorted, color="skyblue")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.title("Importancia de las características - Random Forest")
plt.gca().invert_yaxis()  # Invertir el orden para que la más importante esté arriba
plt.tight_layout()
plt.show()

# Regresión Logística
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train, y_train)

# Extraección de coeficientes
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': logistic_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print(coefficients)

y_pred_logistic = logistic_model.predict(X_test)
print(classification_report(y_test, y_pred_logistic))
print(f"Precisión de Regresión Logística: {accuracy_score(y_test, y_pred_logistic)}")


# Nuevo modelo: XGBoost
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Hacer predicciones
y_pred_xgb = xgb_model.predict(X_test)

# Evaluar el modelo
print("Evaluación del modelo XGBoost:")
print(classification_report(y_test, y_pred_xgb))
print(f"Precisión XGBoost: {accuracy_score(y_test, y_pred_xgb)}")

# SHAP para XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Gráfica de SHAP
shap.summary_plot(shap_values, X_test)

# Optimización de hiperparámetros 
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(xgb.XGBClassifier(random_state=42), param_grid, cv=3, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
print(f"Mejor puntuación: {grid_search.best_score_}")

# Entrenar el modelo XGBoost con los mejores parámetros
best_params = grid_search.best_params_

# Reentrenar el modelo con los mejores parámetros
xgb_model_best = xgb.XGBClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    random_state=42
)

# Entrenar el modelo con el conjunto de entrenamiento
xgb_model_best.fit(X_train, y_train)

# Hacer predicciones con el modelo entrenado
y_pred_best = xgb_model_best.predict(X_test)

# Evaluación del modelo
print("Evaluación del modelo XGBoost con mejores parámetros:")
print(classification_report(y_test, y_pred_best))
print(f"Precisión XGBoost con mejores parámetros: {accuracy_score(y_test, y_pred_best)}")

# Evaluación con AUC y matriz de confusión
auc = roc_auc_score(y_test, y_pred_best)
print(f"AUC: {auc}")

cm = confusion_matrix(y_test, y_pred_best)
print("Matriz de confusión:")
print(cm)

fpr, tpr, thresholds = roc_curve(y_test, xgb_model_best.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, color='b', label='AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Validación cruzada
scores = cross_val_score(xgb_model_best, X_train, y_train, cv=5, scoring='accuracy')
print(f"Precisión media: {scores.mean()} ± {scores.std()}")

# SHAP para XGBoost
explainer = shap.TreeExplainer(xgb_model_best)
shap_values = explainer.shap_values(X_test)

# Gráfico de resumen de SHAP
shap.summary_plot(shap_values, X_test)

# Predicciones de riesgo de desarrollar la enfermedad.
y_pred_proba = xgb_model_best.predict_proba(X_test)
probabilidades_riesgo = y_pred_proba[:, 1]

def clasificar_riesgo(probabilidad):
    if probabilidad < 0.4:
        return "Bajo riesgo"
    elif probabilidad >= 0.4 and probabilidad < 0.7:
        return "Riesgo moderado"
    else:
        return "Alto riesgo"

print("\nPrimeras 10 predicciones de riesgo de Parkinson:")
for i in range(10):  # Mostrar las primeras 10 predicciones
    riesgo = clasificar_riesgo(probabilidades_riesgo[i])
    print(f"ID {i}: Probabilidad de riesgo {probabilidades_riesgo[i]:.2f}, Clasificación: {riesgo}")







#Graficado importancia

import pandas as pd

# Crear un diccionario con las características y sus coeficientes
data = {
    "Feature": [
        "Tremor", "PosturalInstability", "Bradykinesia", "Rigidity", 
        "Depression", "TraumaticBrainInjury", "Stroke", "Diabetes", 
        "FamilyHistoryParkinsons", "Gender", "Constipation", "UPDRS", 
        "Hypertension", "Age", "AlcoholConsumption", "BMI", 
        "CholesterolLDL", "CholesterolTriglycerides", "CholesterolTotal", 
        "DiastolicBP", "CholesterolHDL", "SystolicBP", "Smoking", 
        "PhysicalActivity", "Ethnicity", "DietQuality", "SpeechProblems", 
        "EducationLevel", "SleepQuality", "MoCA", "SleepDisorders", 
        "FunctionalAssessment"
    ],
    "Coefficient": [
        2.136273, 1.877465, 1.791235, 1.697003, 0.532302, 0.494529, 0.448398, 
        0.237521, 0.146557, 0.141311, 0.055523, 0.025618, 0.017779, 0.016308, 
        0.011848, 0.007451, 0.000282, 0.000130, -0.002262, -0.002301, -0.002444, 
        -0.005263, -0.007069, -0.014962, -0.029375, -0.032552, -0.045587, 
        -0.046019, -0.066308, -0.070376, -0.274126, -0.321826
    ]
}

# Convertir el diccionario a un DataFrame
df = pd.DataFrame(data)

# Ordenar por el valor del coeficiente (de mayor a menor)
df_sorted = df.sort_values(by="Coefficient", ascending=False)

# Mostrar la tabla
print(df_sorted)
