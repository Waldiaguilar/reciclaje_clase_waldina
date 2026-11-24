
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# Configuración de la página
st.set_page_config(
    page_title="Clasificador inteligente de residuos",
    page_icon="♻️",
    layout="centered"
)

# Rutas de los archivos
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelo_reciclaje_mobilenet"

MODEL_PATH_H5 = MODEL_DIR / "waste_mobilenet.h5"
MODEL_PATH_KERAS = MODEL_DIR / "waste_mobilenet.keras"
CLASSES_PATH = MODEL_DIR / "class_names.json"

IMG_SIZE = (224, 224)

# Traducción de las categorías
LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura general"
}

# Recomendaciones de reciclaje
RECOMMENDATIONS = {
    "cardboard": "Deposita el cartón limpio y seco en el contenedor de reciclaje.",
    "glass": "Separa el vidrio y deposítalo en un contenedor adecuado.",
    "metal": "Limpia el metal antes de depositarlo para su reciclaje.",
    "paper": "Mantén el papel limpio, seco y separado de otros residuos.",
    "plastic": "Lava el plástico y verifica si es reciclable.",
    "trash": "Deposita los residuos no reciclables en el contenedor correspondiente."
}


@st.cache_data
def load_class_names():
    if not CLASSES_PATH.exists():
        st.error("No se encontró el archivo class_names.json.")
        st.stop()

    try:
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = json.load(f)

        if not isinstance(classes, list) or not classes:
            st.error("El archivo de clases no tiene un formato válido.")
            st.stop()

        return classes

    except Exception as e:
        st.error(f"Error al cargar las clases: {e}")
        st.stop()


@st.cache_resource
def load_model():
    if MODEL_PATH_KERAS.exists():
        model_path = MODEL_PATH_KERAS
    elif MODEL_PATH_H5.exists():
        model_path = MODEL_PATH_H5
    else:
        st.error("No se encontró el modelo de inteligencia artificial.")
        st.stop()

    try:
        return tf.keras.models.load_model(
            str(model_path),
            compile=False
        )

    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        st.stop()


# Procesamiento corregido de imágenes
def prepare_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)

    # MobileNetV2 ya realiza la normalización internamente
    arr = np.asarray(image, dtype=np.float32)

    return np.expand_dims(arr, axis=0)


def predict_image(model, image):
    x = prepare_image(image)
    predictions = model.predict(x, verbose=0)[0]
    return predictions


# Interfaz principal
st.title("♻️ Clasificador inteligente de residuos")

st.write(
    "Identifica diferentes tipos de residuos mediante "
    "inteligencia artificial y contribuye al cuidado del medio ambiente."
)

class_names = load_class_names()
model = load_model()

categorias = ", ".join(
    LABELS_ES.get(label, label) for label in class_names
)

st.caption(f"Categorías: {categorias}.")

st.divider()

# Selección del método de captura
st.subheader("📷 Captura de imágenes")

opcion = st.radio(
    "Selecciona el método de captura:",
    ["Tomar fotografía", "Subir imagen"],
    horizontal=True
)

if opcion == "Tomar fotografía":
    uploaded_file = st.camera_input(
        "Toma una fotografía del residuo"
    )
else:
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png"]
    )


# Clasificación de la imagen
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(
            image,
            caption="Imagen analizada",
            use_container_width=True
        )

        with st.spinner("Analizando el residuo..."):
            predictions = predict_image(model, image)

        if len(predictions) != len(class_names):
            st.error("Las clases no coinciden con las salidas del modelo.")
            st.stop()

        top3_idx = np.argsort(predictions)[-3:][::-1]

        principal_idx = int(top3_idx[0])
        raw_label = class_names[principal_idx]
        label_es = LABELS_ES.get(raw_label, raw_label)
        confidence = float(predictions[principal_idx]) * 100

        st.divider()

        st.subheader("🔍 Resultado de la clasificación")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Residuo identificado", label_es)

        with col2:
            st.metric(
                "Confianza del modelo",
                f"{confidence:.2f}%"
            )

        if confidence < 60:
            st.warning(
                "La predicción tiene baja confianza. "
                "Intenta utilizar una imagen más clara."
            )
        else:
            st.success(f"Predicción principal: {label_es}")

        st.subheader("🌱 Recomendación de reciclaje")

        st.info(
            RECOMMENDATIONS.get(
                raw_label,
                "Verifica cómo debe reciclarse este residuo."
            )
        )

        st.subheader("🏆 Tres posibles resultados")

        for idx in top3_idx:
            label = class_names[idx]
            label = LABELS_ES.get(label, label)
            score = float(predictions[idx]) * 100

            st.write(f"**{label}:** {score:.2f}%")

        st.subheader("📊 Probabilidad por categoría")

        for i, raw_label in enumerate(class_names):
            label = LABELS_ES.get(raw_label, raw_label)
            probability = float(predictions[i])

            st.progress(
                min(max(probability, 0.0), 1.0),
                text=f"{label}: {probability * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")

else:
    st.info("Toma una fotografía o sube una imagen para comenzar.")

st.divider()

st.caption(
    "Proyecto de inteligencia artificial aplicado al reciclaje | UTH"
)
