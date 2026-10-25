# YouTube Performance NLP

Proyecto desarrollado para la asignatura **Procesamiento de Lenguaje Natural** de la Especialización en Inteligencia Artificial de la Universidad de Cundinamarca.

## Objetivo

Desarrollar un sistema de Procesamiento de Lenguaje Natural capaz de analizar comentarios de videos de YouTube y estimar su nivel de desempeño.

## Arquitectura

Comentarios de YouTube → Limpieza NLP → Linear SVC → Análisis de sentimiento → Agregación por video → TF-IDF + variables estadísticas → Logistic Regression → Desempeño alto / bajo.

## Modelos

### Análisis de sentimiento

Se utilizó Linear SVC optimizado mediante GridSearchCV.

- Macro F1 validación cruzada: 0.6478
- Macro F1 prueba: 0.6471
- Balanced Accuracy prueba: 0.6563

### Clasificación de desempeño

Se utilizó Logistic Regression.

Resultados con sentimiento automático:

- Accuracy: 0.8155
- F1-score: 0.8150
- ROC-AUC: 0.8952

## Variables utilizadas

- `Comments_Text`
- `Log_Comment_Likes_Mean`
- `Positive_Ratio_Pred`
- `Negative_Ratio_Pred`
- `Avg_Comment_Words`

## Interfaz

La aplicación está desarrollada con Streamlit. Permite ingresar comentarios manualmente o mediante CSV, analizar su sentimiento y estimar el desempeño alto o bajo del video.

## Dataset

YouTube Statistics - Kaggle.

Archivos utilizados:

- `videos-stats.csv`
- `comments.csv`

## Tecnologías

Python, Google Colab, pandas, NumPy, scikit-learn, TF-IDF, Streamlit y joblib.

## Autor

Pedro Pascual Murcia Vargas  
Universidad de Cundinamarca

## Nota metodológica

Los comentarios son señales posteriores a la publicación del video. Por esta razón, el resultado se interpreta como una estimación del desempeño observado a partir de la reacción de la audiencia y no como un pronóstico previo a la publicación.


## Reproducibilidad y ejecución local

1. Instalar una versión compatible de Python (la prueba de Colab registró Python 3.13.16).
2. Instalar las dependencias fijadas con `pip install -r requirements.txt`.
3. Confirmar que `modelo_nlp_youtube_avance3.pkl` está en la misma carpeta que `app.py`.
4. Ejecutar `streamlit run app.py`.
5. Ingresar comentarios manualmente con sus likes o cargar un CSV que contenga `Comment` y opcionalmente `Likes`.

El modelo clasifica el **desempeño observado**, no pronostica visualizaciones antes de la publicación.

## Cuaderno Google Colab y evaluación (Avance IV)

[Notebook principal Colabyoutube](https://colab.research.google.com/drive/1Up3pJZOrBTCWimw5BpdBKC8K31Uvtalr)

La evaluación independiente en el notebook registra:
- Sentimiento con Linear SVC: Macro F1 test de **0.6471** y Balanced Accuracy de **0.6563**.
- Desempeño con Logistic Regression y sentimiento automático: F1 test de **0.8150** y ROC-AUC test de **0.8952**, evaluado sobre 374 videos.
- Ablación: F1 CV de **0.8120** sin sentimiento y **0.8184** con sentimiento automático.

Se utilizaron particiones por Video ID y predicciones Out-of-Fold para reducir el riesgo de fuga de información entre comentarios del mismo video. La aplicación se probó localmente y mediante un túnel público temporal.

## Aplicación en Streamlit Community Cloud

URL configurada: **https://youtubenlp.streamlit.app/**

**Estado al 25 de octubre de 2026:** el despliegue permanente aún requiere corregir/validar el entorno. La instancia mostraba `ModuleNotFoundError` al importar `joblib` y estaba usando Python 3.14.9. Se recomienda comprobar los logs completos y redeplegar con Python **3.13**, conservando el entorno de bibliotecas probado en Colab. Esta URL se incluye como destino del despliegue, no como certificación de que está operativa.

[Guía oficial de Streamlit para cambiar Python en Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).

## Alcance y limitaciones

- El modelo opera sobre una muestra del dataset YouTube Statistics de Kaggle; no cubre necesariamente todos los comentarios reales de cada video.
- El procesamiento fue diseñado principalmente para comentarios en inglés.
- Los indicadores offline no sustituyen pruebas de generalización en producción.
- El proyecto académico se desarrolló de forma **individual**.
