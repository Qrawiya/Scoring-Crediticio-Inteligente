# 🧠 Modelos DNN y ResNet para Clasificación de Riesgo Crediticio

Este proyecto implementa y compara redes neuronales profundas (DNN) y redes neuronales tipo ResNet para predecir el riesgo crediticio de clientes en base a sus características financieras.

## 📁 Contenido del Proyecto

- **Modelos implementados:**
  - DNN Simple
  - ResNet Simple
  - DNN con Hiperparámetros Optimizado
  - ResNet Optimizado y Ajustado

- **Métricas evaluadas:**
  - Accuracy
  - AUC (Área Bajo la Curva ROC)
  - Matriz de Confusión
  - Precision, Recall y F1-Score
  - Análisis de errores Tipo I y II
  - Supuesto de Análisis Financiero

---

## 📊 Resultados Comparativos

| Modelo                   | Accuracy | AUC     |
|--------------------------|----------|---------|
| DNN Simple               | 0.8429   | 0.8997  |
| ResNet Simple            | 0.8071   | 0.8805  |
| DNN Optimizado           | 0.8286   | 0.9110  |
| ResNet Optimizado        | 0.8321   | 0.8970  |

---

## 🔍 Reflexión Crítica

Se consideran aspectos éticos como el posible sesgo en los datos históricos de aprobación/rechazo. Además, se evalúa la interpretabilidad de los modelos para su aplicación en un contexto real, como en un comité de riesgo bancario. A pesar de usar redes complejas, se acompaña con métricas claras para comunicar los resultados.

---

## 💰 Suposición Financiera (Ejemplo)

- **Falso Positivo (FP):** Otorgar crédito a un cliente riesgoso → pérdida de $5,000
- **Falso Negativo (FN):** Rechazar a un cliente bueno → pérdida de $1,000

Esto permite estimar el impacto financiero total de las predicciones erróneas por modelo y sugiere que, en un contexto real, además de accuracy, el costo financiero de los errores debe considerarse al seleccionar el modelo más adecuado.

---

## 📦 Requisitos

- Python ≥ 3.8
- TensorFlow ≥ 2.13
- Optuna (para optimización)
- scikit-learn
- matplotlib / seaborn

```bash
pip install tensorflow optuna scikit-learn matplotlib seaborn

📌 Autor

Proyecto desarrollado por [Cristobal Araya] como parte de una evaluación de modelos de machine learning aplicados al crédito. Demostración de habilidades en:

Preprocesamiento y modelado

Optimización de hiperparámetros

Interpretación de resultados

Ética y negocio en IA

Reflexión sobre el uso de herramientas de asistencia IA

Este proyecto fue desarrollado con una combinación de criterio propio, conocimiento técnico y asistencia con herramientas basadas en inteligencia artificial (como ChatGPT).

Si bien parte del código fue co-creado con soporte de IA, todas las decisiones de diseño, evaluación, ajustes de modelos, interpretación de resultados y reflexiones fueron definidas por mí como autor del proyecto.

Este enfoque refleja una práctica profesional moderna en la ciencia de datos, donde se utilizan herramientas avanzadas para mejorar la productividad y el enfoque analítico, sin reemplazar el juicio humano.

Subo este proyecto con el objetivo de compartir el proceso completo, desde el desarrollo hasta la reflexión crítica, incluyendo tanto los aspectos técnicos como éticos del modelado de riesgos en un contexto financiero.

## 🚀 Ideas para Mejorar el Proyecto

Si estás interesado en seguir mejorando este proyecto o aprendiendo nuevas herramientas, aquí tienes algunas sugerencias prácticas que puedes implementar. Cada una incluye una breve explicación y cómo comenzar a aplicarla.

---

### 📌 1. Agregar Interpretabilidad al Modelo

**¿Qué hacer?**  
Implementa herramientas de interpretabilidad como **SHAP** o **LIME** para entender cómo cada variable influye en las predicciones del modelo.

**¿Por qué es útil?**  
Ayuda a comunicar tus resultados a personas no técnicas (como un equipo de negocio o comité de crédito) y mejora la confianza en el modelo.

**¿Cómo empezar?**  
Instala e implementa alguna de estas librerías:

```bash
pip install shap lime
```

Consulta la documentación de [`shap`](https://github.com/slundberg/shap) y [`lime`](https://github.com/marcotcr/lime) para ver ejemplos aplicados.

---

### 📌 2. Visualizar la Importancia de las Variables

**¿Qué hacer?**  
Genera gráficos que muestren qué variables tienen más influencia en el modelo, usando:

- Permutation Importance  
- SHAP summary plots

**¿Por qué es útil?**  
Te permite justificar decisiones del modelo y hacer análisis de negocio más claro.

**¿Cómo empezar?**  
Usa las funciones de `shap` o librerías como `eli5` para visualizaciones simples:

```bash
pip install shap eli5
```

---

### 📌 3. Usar Cross-Validation en la Optimización

**¿Qué hacer?**  
En lugar de usar solo un `train/test split`, implementa **k-fold cross-validation** para entrenar y evaluar el modelo.

**¿Por qué es útil?**  
Aumenta la robustez de tus resultados y evita que un split afortunado afecte las métricas.

**¿Cómo empezar?**  
Si ya estás usando `Optuna` o `scikit-learn`, puedes integrar fácilmente `StratifiedKFold` en tu pipeline:

```python
from sklearn.model_selection import StratifiedKFold
```

---

### 📌 4. Comparar con Modelos Tradicionales

**¿Qué hacer?**  
Agrega un modelo de referencia simple, como `LogisticRegression`, como punto de comparación (**baseline**).

**¿Por qué es útil?**  
A veces los modelos simples ofrecen buen rendimiento y mejor interpretabilidad, lo cual es clave en escenarios reales.

**¿Cómo empezar?**  
Usa este modelo con `scikit-learn`:

```bash
pip install scikit-learn
```

```python
from sklearn.linear_model import LogisticRegression
```

---

### 📌 5. Crear un Dashboard con Streamlit

**¿Qué hacer?**  
Construye una app web interactiva para que otros puedan probar el modelo cargando sus propios datos.

**¿Por qué es útil?**  
Ideal para entrevistas, mostrar el proyecto en tu portafolio, o presentarlo a usuarios no técnicos.

**¿Cómo empezar?**

1. Instala Streamlit:
   ```bash
   pip install streamlit
   ```
2. Crea un archivo `app.py` con tu lógica de predicción.
3. Ejecuta la app:
   ```bash
   streamlit run app.py
   ```

---

### 📌 6. Agregar Tests y Validaciones de Calidad

**¿Qué hacer?**  
Incluye pequeñas pruebas que aseguren la calidad de los datos de entrada y salida, como:

- Verificar que no haya valores nulos (`NaN`)
- Validar consistencia de las columnas
- Chequear fairness del modelo en distintos grupos

**¿Por qué es útil?**  
Da mayor confianza en el modelo y muestra profesionalismo en el desarrollo.

**¿Cómo empezar?**  
Puedes usar `pytest`, o simplemente agregar bloques `assert` en el código:

```python
assert not df.isnull().any().any(), "Existen valores nulos en los datos"
```

---

