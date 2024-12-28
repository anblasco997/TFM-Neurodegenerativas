

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#Tratamiento de datos

# 1. Carga de datos 
ruta_archivo = "C:\\Users\\Antonio\\Downloads\\alzheimer.csv"
data = pd.read_csv(ruta_archivo)

# 2. Identificar y eliminar valores nulos
print("Valores nulos por columna:")
print(data.isnull().sum())

# Eliminar  valores nulos
data = data.dropna()  # Esto elimina filas con valores nulos (ajusta según el caso)

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

# Reducción de dimensionalidad
threshold = 0.01  # Umbral de importancia
selected_features = feature_names[importances > threshold]

# Redefinir los datos
X_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]

# Entrenar el modelo con datos reducidos
rf_model_reduced = RandomForestClassifier(random_state=42)
rf_model_reduced.fit(X_reduced, y_train)

# Evaluar el modelo reducido
y_pred_reduced = rf_model_reduced.predict(X_test_reduced)
print("Evaluación tras reducción de dimensionalidad:")
print(classification_report(y_test, y_pred_reduced))
print(f"Precisión: {accuracy_score(y_test, y_pred_reduced)}")

# Regresión Logística
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_reduced, y_train)

# Extraección de coeficientes
coefficients = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': logistic_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print(coefficients)

y_pred_logistic = logistic_model.predict(X_test_reduced)
print(classification_report(y_test, y_pred_logistic))
print(f"Precisión de Regresión Logística: {accuracy_score(y_test, y_pred_logistic)}")

# SHAP para RandomForest y encontrar los mejores parámetros o factores
explainer = shap.TreeExplainer(rf_model_reduced)
shap_values = explainer.shap_values(X_test_reduced)


if isinstance(shap_values, list):
    shap_values_class_1 = shap_values[1]  # Clase positiva (1)
else:
    shap_values_class_1 = shap_values  # Para otros modelos que devuelvan directamente un array

# Gráfica de SHAP
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_class_1, X_test_reduced, plot_type="bar", max_display=10)

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

print("\nPrimeras 10 predicciones de riesgo de Alzheimer:")
for i in range(10):  # Mostrar las primeras 10 predicciones
    riesgo = clasificar_riesgo(probabilidades_riesgo[i])
    print(f"ID {i}: Probabilidad de riesgo {probabilidades_riesgo[i]:.2f}, Clasificación: {riesgo}")










#Parkinson







