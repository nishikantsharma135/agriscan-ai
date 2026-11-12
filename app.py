
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AgriScan AI",
    page_icon="🌱",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.main {
    background-color: #f5f9f4;
}
.title {
    color: #176b3a;
    font-size: 42px;
    font-weight: bold;
}
.subtitle {
    color: #526b59;
    font-size: 18px;
}
.result {
    padding: 20px;
    border-radius: 12px;
    background-color: #e6f4e9;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD AI MODEL ----------------

@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model(
        "crop_disease_model.keras"
    )

@st.cache_data
def load_labels():
    with open("class_names.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]

# ---------------- DISEASE INFORMATION ----------------

disease_info = {
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "description": "A fungal disease that causes dark brown spots on potato leaves.",
        "recommendations": [
            "Remove infected leaves.",
            "Avoid overhead irrigation.",
            "Maintain proper plant spacing.",
            "Consult an agricultural expert about suitable fungicides."
        ]
    },

    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "description": "A fungal disease that causes brown spots with concentric rings on tomato leaves.",
        "recommendations": [
            "Remove infected plant material.",
            "Avoid wetting leaves during irrigation.",
            "Maintain good air circulation.",
            "Consult an agricultural expert about disease management."
        ]
    },

    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "description": "A serious disease that can cause dark lesions and rapid plant damage.",
        "recommendations": [
            "Remove severely infected plant material safely.",
            "Avoid prolonged leaf wetness.",
            "Monitor nearby plants.",
            "Seek prompt advice from an agricultural expert."
        ]
    },

    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "description": "The model classified this image as a healthy tomato leaf.",
        "recommendations": [
            "Continue regular crop monitoring.",
            "Maintain balanced irrigation.",
            "Keep the field clean.",
            "Monitor leaves regularly for new symptoms."
        ]
    }
}

# ---------------- HEADER ----------------

st.markdown(
    '<p class="title">🌱 AgriScan AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-Powered Crop Disease Detection and Agricultural Risk Assessment</p>',
    unsafe_allow_html=True
)

st.divider()

st.success(
    "Welcome to AgriScan AI — Your smart agricultural assistant!"
)

# ---------------- NAVIGATION ----------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "AI Disease Detection",
        "Environmental Risk Calculator",
        "About Project"
    ]
)

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.header("Welcome to AgriScan AI")

    st.write(
        "AgriScan AI is an artificial intelligence based "
        "agricultural website designed to identify potential "
        "crop diseases using leaf images."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        ### 🔍 AI Disease Detection

        Upload a crop leaf photograph and let our trained "
        "MobileNetV2 model predict its disease category.
        """)

    with col2:
        st.info("""
        ### 🌦️ Environmental Risk Calculator

        Enter temperature, humidity and leaf wetness "
        "information to estimate environmental disease risk.
        """)

    st.subheader("Supported Categories")

    st.write("""
    - Potato Early Blight
    - Tomato Early Blight
    - Tomato Late Blight
    - Healthy Tomato
    """)

    st.warning(
        "This project is an educational prototype. "
        "Its predictions should not replace expert diagnosis."
    )

# ---------------- AI DISEASE DETECTION ----------------

elif page == "AI Disease Detection":

    st.header("🌿 AI Crop Disease Detection")

    st.write(
        "Upload a clear photograph of a potato or tomato leaf."
    )

    uploaded_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Crop Leaf",
            width=350
        )

        if st.button("Detect Crop Disease", type="primary"):

            with st.spinner("Analyzing leaf image..."):

                try:
                    model = load_ai_model()
                    class_names = load_labels()

                    image = image.resize((224, 224))

                    image_array = np.array(image)
                    image_array = np.expand_dims(
                        image_array, axis=0
                    )

                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )

                    predicted_index = int(
                        np.argmax(predictions[0])
                    )

                    confidence = float(
                        predictions[0][predicted_index]
                    ) * 100

                    predicted_class = class_names[
                        predicted_index
                    ]

                    details = disease_info.get(
                        predicted_class
                    )

                    if details is None:
                        st.error(
                            "Disease information not available."
                        )

                    else:

                        st.divider()

                        st.subheader("Prediction Results")

                        st.markdown(
                            f"""
                            <div class="result">
                            <h2>{details['name']}</h2>
                            <h3>Confidence: {confidence:.2f}%</h3>
                            <p>{details['description']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.subheader(
                            "Disease Management Recommendations"
                        )

                        for item in details["recommendations"]:
                            st.write("✅ " + item)

                        st.caption(
                            "Confidence is the model's classification "
                            "score, not a guarantee of diagnostic accuracy."
                        )

                except Exception as e:
                    st.error(
                        f"Prediction error: {e}"
                    )

# ---------------- ENVIRONMENTAL RISK CALCULATOR ----------------

elif page == "Environmental Risk Calculator":

    st.header("🌦️ Environmental Crop Risk Calculator")

    st.write(
        "Enter environmental conditions to estimate "
        "potential disease-favouring conditions."
    )

    temperature = st.slider(
        "Temperature (°C)",
        min_value=0,
        max_value=50,
        value=25
    )

    humidity = st.slider(
        "Relative Humidity (%)",
        min_value=0,
        max_value=100,
        value=60
    )

    leaf_wetness = st.selectbox(
        "Leaf Wetness",
        ["Dry", "Moderately Wet", "Wet"]
    )

    symptoms = st.checkbox(
        "Visible spots or unusual leaf discoloration"
    )

    if st.button("Calculate Environmental Risk"):

        risk = 0

        if 20 <= temperature <= 30:
            risk += 20

        if humidity >= 80:
            risk += 35
        elif humidity >= 60:
            risk += 20

        if leaf_wetness == "Wet":
            risk += 30
        elif leaf_wetness == "Moderately Wet":
            risk += 15

        if symptoms:
            risk += 15

        risk = min(risk, 100)

        st.subheader("Risk Assessment")

        st.metric(
            "Estimated Risk Score",
            f"{risk}/100"
        )

        st.progress(risk / 100)

        if risk < 30:
            st.success("Low Environmental Risk")

        elif risk < 60:
            st.warning("Moderate Environmental Risk")

        else:
            st.error("High Environmental Risk")

        st.caption(
            "This is an illustrative rule-based educational score, "
            "not a validated disease prediction or probability. "
            "It does not account for crop-specific disease models."
        )

        st.info(
            "Continue monitoring your crops and consult an "
            "agricultural expert if symptoms appear."
        )

# ---------------- ABOUT PAGE ----------------

elif page == "About Project":

    st.header("About AgriScan AI")

    st.write("""
    AgriScan AI is a first-year engineering AI practical project.

    Its main objective is to demonstrate how artificial intelligence
    can assist in preliminary crop disease screening.
    """)

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - TensorFlow and Keras
    - MobileNetV2
    - PlantVillage Dataset
    - Google Colab
    - GitHub
    - Streamlit
    """)

    st.subheader("Project Modules")

    st.write("""
    1. Machine-learning based leaf image classification.
    2. Rule-based environmental risk calculator.
    """)

    st.success(
        "Developed as an Artificial Intelligence Practical Project."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AgriScan AI | AI-Based Crop Disease Detection System | "
    "Educational Project"
)
