<img width="1280" height="640" alt="WhatsApp Image 2026-11-17 at 3 00 46 PM (12)" src="https://github.com/user-attachments/assets/b15d4742-6d2b-44eb-91bb-d94161e10c65" />

# 🛡️ Ν Proyecto Nu | Predicción de Beneficios de Seguros y Protección de Datos (Sure Tomorrow)

## 📋 Descripción
Las compañías de seguros administran información personal, familiar y financiera clave para comprender mejor a sus clientes, identificar perfiles similares y desarrollar modelos predictivos que apoyen la toma de decisiones estratégicas.

En este proyecto se analizan datos históricos de 5,000 clientes de la aseguradora **Sure Tomorrow** mediante técnicas de **Machine Learning**, algoritmos de **vecinos más cercanos (k-NN)** y **álgebra lineal**, abordando cuatro necesidades críticas de negocio:
1. Encontrar clientes con características similares.
2. Predecir si un cliente recibirá beneficios de seguro.
3. Estimar la cantidad de beneficios que podría recibir.
4. Proteger los datos personales mediante ofuscación matricial sin afectar el desempeño predictivo.

## 🎯 Objetivos
- ✅ Examinar la estructura, calidad y tipos de datos del conjunto de 5,000 clientes.
- ✅ Encontrar clientes similares mediante el algoritmo **k-Nearest Neighbors (k-NN)** analizando el impacto del escalado (`MaxAbsScaler`) y las métricas de distancia (Euclidiana vs. Manhattan).
- ✅ Construir una variable objetivo binaria y entrenar un modelo de clasificación evaluado con la métrica **F1 Score** ($k=1\dots10$) frente a modelos aleatorios de referencia (*Dummy*).
- ✅ Implementar una **Regresión Lineal personalizada** desde cero mediante álgebra matricial para predecir la cantidad de beneficios, evaluada con **RMSE** y **$R^2$**.
- ✅ Desarrollar un método de **ofuscación de datos** con matrices invertibles, demostrando analítica y computacionalmente que conserva intacta la calidad del modelo.

## 🛠️ Herramientas y Habilidades Utilizadas
* **Lenguaje:** Python
* **Librerías:** Pandas, NumPy, Scikit-Learn y Seaborn
* **Entorno:** Jupyter Notebook (`Nu___Insurance_Benefits_Prediction_and_Data_Protection.ipynb`)
* **Técnicas:** Vecinos Más Cercanos (k-NN), Escalado de características (`MaxAbsScaler` y `StandardScaler`), Clasificación supervisada, Álgebra Lineal Matricial, Regresión Lineal y Ofuscación de Datos.

## 📊 Fuente de Datos y Variables
El análisis utiliza el dataset histórico `insurance_us.csv` (5,000 registros sin valores nulos):
| Variable | Descripción |
| :--- | :--- |
| `gender` | Género registrado del cliente (`0` o `1`) |
| `age` | Edad del cliente (convertida a entero) |
| `income` | Ingresos anuales estimados |
| `family_members` | Número de familiares del cliente |
| `insurance_benefits` | Cantidad de beneficios de seguro recibidos (variable objetivo) |

## ⚙️ Metodología y Resultados Obtenidos

### 1. Tarea 1: Búsqueda de Clientes Similares (k-NN)
Aplicación de *NearestNeighbors* comparando datos originales vs. datos escalados (`MaxAbsScaler`) bajo distancias **Euclidiana** y **Manhattan**:
* **Hallazgo clave:** Sin escalado, la variable `income` (decenas de miles) domina por completo el cálculo de distancias. Al escalar los datos, los vecinos seleccionados cambian por completo (de `[0, 2022, 1225, 4031, 3424]` a `[0, 2689, 133, 4869, 3365]`), permitiendo que la edad, el género y el tamaño familiar participen equilibradamente.

### 2. Tarea 2: Predicción de Recepción de Beneficios (Clasificación)
Transformación de `insurance_benefits` en una variable binaria (`0`: 4,436 clientes sin beneficios; `1`: 564 clientes con beneficios) y partición **70:30** (`random_state=12345`):
* **Modelo Aleatorio (Dummy):** Alcanzó un techo máximo de referencia de **F1 = 0.20** (para probabilidades $p=0.50$ y $p=1.00$).
* **kNN sin escalar:** Desempeño deficiente e inestable que cae desde **F1 = 0.6071** ($k=1$) hasta **F1 = 0.0000** ($k=10$).
* **kNN con datos escalados (`StandardScaler`):** Rendimiento sobresaliente, alcanzando un pico máximo de **F1 = 0.9571** ($k=1$) y manteniéndose por encima de **0.88** para todos los valores de $k$ hasta 10.

### 3. Tarea 3: Estimación de Beneficios (Regresión Lineal Matricial)
Implementación de la clase `MyLinearRegression` mediante operaciones de álgebra lineal:
$$w = (X^T X)^{-1} X^T y \qquad \hat{y} = Xw$$
* **Resultados (Original vs. Escalado):** Tanto con los datos originales como con los escalados, el modelo obtuvo exactamente las mismas métricas: **RMSE = 0.34** y **$R^2$ = 0.66**, demostrando que el escalado ajusta proporcionalmente los pesos sin alterar las predicciones.

### 4. Tarea 4: Protección de Datos Personales (Ofuscación Matricial)
Transformación de la matriz de características $X$ multiplicándola por una matriz aleatoria invertible $P$:
$$X' = X P \qquad \text{Recuperación:} \quad X = X' P^{-1}$$
Dado que los pesos del modelo transformado cumplen $w_P = P^{-1}w$, se demostró analíticamente que las predicciones son idénticas:
$$\hat{y}_P = (XP)(P^{-1}w) = X(PP^{-1})w = Xw = \hat{y}$$
* **Validación computacional:** Al entrenar `MyLinearRegression` con los datos ofuscados $X'$, se obtuvieron exactamente las mismas métricas que con los datos originales (**RMSE = 0.34** y **$R^2$ = 0.66**), garantizando privacidad sin pérdida de calidad predictiva.

## 📂 Archivos Principales
* `Nu___Insurance_Benefits_Prediction_and_Data_Protection.ipynb`
* `insurance_us.csv`

## 🏁 Conclusión
El proyecto demostró cómo integrar **Machine Learning** y **álgebra lineal** para resolver múltiples retos operativos de **Sure Tomorrow**. Desde la identificación precisa de perfiles similares mediante k-NN escalado y la clasificación de probabilidad de beneficios (**F1 = 0.9571**), hasta la regresión lineal matricial con **ofuscación de datos** (**RMSE = 0.34**, **$R^2$ = 0.66**), se comprobó que es totalmente viable proteger la privacidad de los clientes sin comprometer el rendimiento de los modelos predictivos.

## 🏁 Conclusión
El proyecto demostró cómo integrar técnicas de **Machine Learning** y **álgebra lineal** para resolver múltiples retos operativos de la aseguradora **Sure Tomorrow**. Desde la segmentación de clientes similares mediante k-NN escalado y la clasificación de probabilidad de beneficios, hasta la implementación matricial de regresión lineal con **ofuscación de datos**, se comprobó que es totalmente viable proteger la privacidad de la información personal sin comprometer la calidad ni la exactitud de los modelos predictivos.
