import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="YouTube Performance NLP",
    page_icon="▶️",
    layout="wide"
)

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_nlp_youtube_avance3.pkl"

@st.cache_resource
def cargar_artefacto():
    return joblib.load(RUTA_MODELO)

artefacto = cargar_artefacto()
modelo_sentimiento = artefacto["modelo_sentimiento"]
modelo_desempeno = artefacto["modelo_desempeno"]
clases_sentimiento = artefacto["clases_sentimiento"]
umbral_views = artefacto.get("umbral_views", None)

def limpiar_comentario(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def preparar_comentarios(df):
    datos = df.copy()

    if "Comment" not in datos.columns:
        raise ValueError("Los datos deben contener una columna llamada Comment.")

    if "Likes" not in datos.columns:
        datos["Likes"] = 0

    datos["Comment"] = datos["Comment"].fillna("").astype(str)
    datos["Likes"] = (
        pd.to_numeric(datos["Likes"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    datos["Comment_Clean"] = datos["Comment"].apply(limpiar_comentario)
    datos = datos[datos["Comment_Clean"].str.len() > 0].copy()
    return datos

def analizar_video(df):
    comentarios = preparar_comentarios(df)

    if len(comentarios) == 0:
        raise ValueError("Debe ingresar al menos un comentario válido.")

    pred_sentimiento = modelo_sentimiento.predict(
        comentarios["Comment_Clean"].tolist()
    )

    comentarios["Sentiment_Pred"] = pred_sentimiento
    comentarios["Sentimiento"] = [
        clases_sentimiento[int(valor)] for valor in pred_sentimiento
    ]

    positive_ratio = (pred_sentimiento == 2).mean()
    neutral_ratio = (pred_sentimiento == 1).mean()
    negative_ratio = (pred_sentimiento == 0).mean()

    comments_text = " ".join(comentarios["Comment_Clean"].tolist())
    avg_comment_words = (
        comentarios["Comment_Clean"].str.split().str.len().mean()
    )
    comment_likes_mean = comentarios["Likes"].mean()
    log_comment_likes_mean = np.log1p(comment_likes_mean)

    entrada_modelo = pd.DataFrame({
        "Comments_Text": [comments_text],
        "Log_Comment_Likes_Mean": [log_comment_likes_mean],
        "Positive_Ratio_Pred": [positive_ratio],
        "Negative_Ratio_Pred": [negative_ratio],
        "Avg_Comment_Words": [avg_comment_words]
    })

    prediccion = int(modelo_desempeno.predict(entrada_modelo)[0])
    probabilidad_alto = float(
        modelo_desempeno.predict_proba(entrada_modelo)[0, 1]
    )

    return {
        "prediccion": prediccion,
        "probabilidad_alto": probabilidad_alto,
        "positive_ratio": float(positive_ratio),
        "neutral_ratio": float(neutral_ratio),
        "negative_ratio": float(negative_ratio),
        "avg_comment_words": float(avg_comment_words),
        "comment_likes_mean": float(comment_likes_mean),
        "log_comment_likes_mean": float(log_comment_likes_mean),
        "comentarios": comentarios,
        "entrada_modelo": entrada_modelo
    }

st.title("▶️ YouTube Performance NLP")

st.write(
    "Aplicación para analizar comentarios de un video de YouTube y estimar "
    "su nivel de desempeño utilizando Procesamiento de Lenguaje Natural."
)

st.info(
    "El sistema analiza automáticamente el sentimiento de los comentarios "
    "y posteriormente utiliza esa información, junto con características "
    "textuales y estadísticas, para clasificar el desempeño del video."
)

st.header("1. Ingreso de comentarios")

modo_entrada = st.radio(
    "Seleccione el método de entrada:",
    ["Ingresar manualmente", "Cargar archivo CSV"],
    horizontal=True
)

if modo_entrada == "Ingresar manualmente":
    st.write(
        "Escriba los comentarios del video. La columna Likes corresponde "
        "a los likes recibidos por cada comentario."
    )

    tabla_inicial = pd.DataFrame({
        "Comment": ["", "", "", "", ""],
        "Likes": [0, 0, 0, 0, 0]
    })

    datos_usuario = st.data_editor(
        tabla_inicial,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Comment": st.column_config.TextColumn("Comentario", width="large"),
            "Likes": st.column_config.NumberColumn(
                "Likes", min_value=0, step=1
            )
        }
    )

else:
    archivo = st.file_uploader("Seleccione un archivo CSV", type=["csv"])

    if archivo is not None:
        datos_usuario = pd.read_csv(archivo)

        if "Comment" not in datos_usuario.columns:
            st.error("El CSV debe contener una columna llamada Comment.")
            st.stop()

        if "Likes" not in datos_usuario.columns:
            datos_usuario["Likes"] = 0
            st.warning(
                "El archivo no contiene la columna Likes. "
                "Se utilizará 0 para esos valores."
            )

        st.write("Vista previa:")
        st.dataframe(
            datos_usuario[["Comment", "Likes"]].head(20),
            use_container_width=True
        )
    else:
        datos_usuario = pd.DataFrame(columns=["Comment", "Likes"])

st.header("2. Ejecutar análisis")

if st.button("🔍 Analizar video", type="primary", use_container_width=True):
    try:
        resultado = analizar_video(datos_usuario)

        st.divider()
        st.header("3. Resultados")
        st.subheader("Análisis de sentimiento")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Comentarios analizados",
            len(resultado["comentarios"])
        )
        col2.metric(
            "Positivos",
            f"{resultado['positive_ratio'] * 100:.1f} %"
        )
        col3.metric(
            "Neutrales",
            f"{resultado['neutral_ratio'] * 100:.1f} %"
        )
        col4.metric(
            "Negativos",
            f"{resultado['negative_ratio'] * 100:.1f} %"
        )

        distribucion = pd.DataFrame({
            "Proporción": [
                resultado["negative_ratio"],
                resultado["neutral_ratio"],
                resultado["positive_ratio"]
            ]
        }, index=["Negativo", "Neutral", "Positivo"])

        st.bar_chart(distribucion)

        st.subheader("Predicción de desempeño")

        if resultado["prediccion"] == 1:
            st.success("📈 DESEMPEÑO ESTIMADO: ALTO")
        else:
            st.warning("📉 DESEMPEÑO ESTIMADO: BAJO")

        probabilidad = resultado["probabilidad_alto"]

        st.metric(
            "Probabilidad de desempeño alto",
            f"{probabilidad * 100:.2f} %"
        )
        st.progress(int(probabilidad * 100))

        if umbral_views is not None:
            umbral_texto = f"{int(umbral_views):,}".replace(",", ".")
            st.caption(
                "En el conjunto de datos utilizado para entrenar el modelo, "
                "desempeño alto corresponde a aproximadamente "
                f"{umbral_texto} visualizaciones o más."
            )

        st.subheader("Detalle de comentarios")

        tabla_resultados = (
            resultado["comentarios"][["Comment", "Likes", "Sentimiento"]]
            .rename(columns={"Comment": "Comentario"})
        )

        st.dataframe(
            tabla_resultados,
            use_container_width=True,
            hide_index=True
        )

        with st.expander("Ver variables calculadas por el modelo"):
            st.write(
                "Promedio de likes por comentario:",
                round(resultado["comment_likes_mean"], 4)
            )
            st.write(
                "Log promedio de likes:",
                round(resultado["log_comment_likes_mean"], 4)
            )
            st.write(
                "Promedio de palabras por comentario:",
                round(resultado["avg_comment_words"], 4)
            )
            st.dataframe(
                resultado["entrada_modelo"],
                use_container_width=True
            )

        st.divider()
        st.caption(
            "Nota: el modelo utiliza comentarios posteriores a la publicación "
            "del video. Por lo tanto, el resultado debe interpretarse como "
            "una estimación del desempeño observado a partir de la reacción "
            "de la audiencia y no como un pronóstico previo a la publicación."
        )

    except Exception as error:
        st.error(f"No fue posible realizar el análisis: {error}")
