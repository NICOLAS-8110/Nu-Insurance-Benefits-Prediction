<img width="1280" height="640" alt="WhatsApp Image 2026-11-17 at 3 00 46 PM (12)" src="https://github.com/user-attachments/assets/b15d4742-6d2b-44eb-91bb-d94161e10c65" />

# 🛡️ Ν Proyecto Nu | Predicción de Beneficios de Seguros y Protección de Datos (Sure Tomorrow)

## 📋 Descripción
Las compañías de seguros administran información personal, familiar y financiera clave para comprender mejor a sus clientes, identificar perfiles similares y desarrollar modelos predictivos que apoyen la toma de decisiones estratégicas.

En este proyecto se analizan datos históricos de clientes de la aseguradora **Sure Tomorrow** mediante técnicas de **Machine Learning**, algoritmos de **vecinos más cercanos (k-NN)** y **álgebra lineal**, abordando cuatro necesidades críticas de negocio:
1. Encontrar clientes con características similares.
2. Predecir si un cliente recibirá beneficios de seguro.
3. Estimar la cantidad de beneficios que podría recibir.
4. Proteger los datos personales mediante ofuscación matricial sin afectar el desempeño predictivo.

## 🎯 Objetivos
- ✅ Examinar la estructura, calidad, valores ausentes y duplicados del conjunto de datos.
- ✅ Encontrar clientes similares mediante el algoritmo **k-Nearest Neighbors (k-NN)** analizando el impacto del escalado y las métricas de distancia (Euclidiana vs. Manhattan).
- ✅ Construir una variable objetivo binaria y entrenar un modelo de clasificación evaluado con la métrica **F1 Score** frente a modelos aleatorios de referencia.
- ✅ Implementar una **Regresión Lineal personalizada** desde cero mediante álgebra matricial para predecir la cantidad de beneficios, evaluada con **RMSE** y **$R^2$**.
- ✅ Desarrollar un método de **ofuscación de datos** con matrices invertibles, demostrando matemáticamente y en la práctica que conserva intacta la calidad del modelo.

## 🛠️ Herramientas y Habilidades Utilizadas
* **Lenguaje:** Python
* **Librerías:** Pandas, NumPy, Scikit-Learn, Matplotlib y Seaborn
* **Entorno:** Jupyter Notebook / Google Colab
* **Técnicas:** Vecinos Más Cercanos (k-NN), Escalado de características (`MaxAbsScaler` / `StandardScaler`), Clasificación supervisada, Álgebra Lineal Matricial, Regresión Lineal y Ofuscación de Datos.

## 📊 Fuente de Datos y Variables
El análisis utiliza el dataset histórico `insurance_us.csv`:
| Variable | Descripción |
| :--- | :--- |
| `gender` | Género registrado del cliente |
| `age` | Edad del cliente |
| `income` | Ingresos anuales estimados |
| `family_members` | Número de familiares del cliente |
| `insurance_benefits` | Cantidad de beneficios de seguro recibidos (variable objetivo) |

## ⚙️ Metodología y Tareas Desarrolladas

### 1. Preparación y Análisis Exploratorio (EDA)
Inspección de dimensiones, normalización de columnas, revisión de duplicados y análisis de distribuciones. Se identificó una fuerte diferencia de escala entre variables (por ejemplo, `income` frente a `age` o `family_members`), factor crítico para modelos basados en distancias.

### 2. Tarea 1: Búsqueda de Clientes Similares (k-NN)
Aplicación del algoritmo *k-Nearest Neighbors* comparando datos originales vs. datos escalados bajo distancias **Euclidiana** y **Manhattan**:
* **Hallazgo clave:** Sin escalado, los ingresos dominan por completo el cálculo de distancias. Al estandarizar los datos, todas las variables participan equilibradamente en la identificación de perfiles similares.

### 3. Tarea 2: Predicción de Recepción de Beneficios (Clasificación)
Transformación de `insurance_benefits` en una variable objetivo binaria (`0` = sin beneficios, `1` = recibió uno o más beneficios) para entrenar un clasificador k-NN evaluado mediante **F1 Score**, superando significativamente el rendimiento de modelos aleatorios de referencia.

### 4. Tarea 3: Estimación de Beneficios (Regresión Lineal Matricial)
Implementación de una clase personalizada de Regresión Lineal utilizando operaciones de álgebra lineal para calcular los pesos ($w$) y predicciones ($\hat{y}$):
$$w = (X^T X)^{-1} X^T y \qquad \hat{y} = Xw$$
Evaluada mediante las métricas **RMSE** (magnitud promedio del error) y **$R^2$** (proporción de variabilidad explicada).

### 5. Tarea 4: Protección de Datos Personales (Ofuscación Matricial)
Transformación de la matriz de características original $X$ multiplicándola por una matriz invertible aleatoria $P$:
$$X' = X P \qquad \text{Recuperación:} \quad X = X' P^{-1}$$
Dado que los pesos del modelo transformado cumplen $w' = P^{-1}w$, se demuestra algebraicamente que las predicciones permanecen idénticas:
$$X'w' = (XP)(P^{-1}w) = X(PP^{-1})w = Xw$$
* **Validación:** La comparación entre datos originales y ofuscados confirmó que tanto las predicciones como los valores de **RMSE** y **$R^2$** se mantienen inalterados, protegiendo la privacidad de los clientes sin sacrificar capacidad analítica.

## 💡 Principales Hallazgos
* **Importancia del escalado:** En algoritmos basados en distancia, omitir la normalización sesga los resultados hacia variables de mayor magnitud numérica como los ingresos.
* **Álgebra lineal aplicada:** Es posible construir modelos de regresión lineal exactos y verificables mediante operaciones matriciales directas.
* **Privacidad sin pérdida de rendimiento:** La transformación mediante matrices invertibles oculta eficazmente la información sensible de los clientes manteniendo intacta la precisión del modelo predictivo.

## 📂 Archivos Principales
* `Nu___Insurance_Benefits_Prediction_and_Data_Protection.ipynb`
* `insurance_us.csv`

## 🏁 Conclusión
El proyecto demostró cómo integrar técnicas de **Machine Learning** y **álgebra lineal** para resolver múltiples retos operativos de la aseguradora **Sure Tomorrow**. Desde la segmentación de clientes similares mediante k-NN escalado y la clasificación de probabilidad de beneficios, hasta la implementación matricial de regresión lineal con **ofuscación de datos**, se comprobó que es totalmente viable proteger la privacidad de la información personal sin comprometer la calidad ni la exactitud de los modelos predictivos.
