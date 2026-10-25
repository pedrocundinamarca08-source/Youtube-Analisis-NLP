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
