import streamlit as st
import pickle
import numpy as np

from src.llm_explainer import explain_prediction
from src.chatbot import ask_medical_assistant

# -------------------------------
# Page Config
# -------------------------------

st.set_page_config(
    page_title="AI Diabetes Assistant",
    page_icon="🏥",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.title("🏥 AI Healthcare Assistant")

st.sidebar.markdown("---")

st.sidebar.write(
    """
    This project combines:

    ✅ Machine Learning

    ✅ Random Forest

    ✅ RAG

    ✅ FAISS

    ✅ Llama 3

    ✅ Streamlit
    """
)

st.sidebar.markdown("---")

st.sidebar.info("Built by Yash Mahure")

# -------------------------------
# Load Model
# -------------------------------

model = pickle.load(
    open("models/diabetes_model.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

# -------------------------------
# Session State
# -------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "explanation" not in st.session_state:
    st.session_state.explanation = None

# -------------------------------
# Main Title
# -------------------------------

st.title(
    "🏥 AI-Powered Diabetes Prediction & Healthcare Assistant"
)

st.markdown(
    "Predict diabetes risk and get AI-powered explanations."
)

# -------------------------------
# Tabs
# -------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "🔮 Prediction",
        "🧠 AI Explanation",
        "💬 Medical Chatbot"
    ]
)

# ====================================================
# TAB 1 - Prediction
# ====================================================

with tab1:

    st.header("Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        preg = st.number_input(
            "Pregnancies",
            min_value=0
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0
        )

        bp = st.number_input(
            "Blood Pressure",
            min_value=0
        )

        skin = st.number_input(
            "Skin Thickness",
            min_value=0
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0
        )

        age = st.number_input(
            "Age",
            min_value=0
        )

    if st.button("Predict"):

        data = np.array([
            [
                preg,
                glucose,
                bp,
                skin,
                insulin,
                bmi,
                dpf,
                age
            ]
        ])

        scaled_data = scaler.transform(data)

        prediction = model.predict(
            scaled_data
        )

        probability = model.predict_proba(
            scaled_data
        )

        confidence = (
            probability.max() * 100
        )

        st.subheader(
            "Prediction Result"
        )

        if prediction[0] == 1:

            result = (
                "Patient is likely diabetic"
            )

            st.error(result)

        else:

            result = (
                "Patient is not diabetic"
            )

            st.success(result)

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.session_state.result = result

        with st.spinner(
            "Generating AI Explanation..."
        ):

            st.session_state.explanation = (
                explain_prediction(
                    prediction=result,
                    glucose=glucose,
                    bmi=bmi,
                    age=age,
                    blood_pressure=bp
                )
            )

    with st.expander(
        "What do these features mean?"
    ):

        st.write(
            """
            **Glucose**
            Blood sugar level.

            **BMI**
            Body Mass Index.

            **DPF**
            Family history score of diabetes.

            **Insulin**
            Amount of insulin in blood.

            **Blood Pressure**
            Patient blood pressure level.
            """
        )

# ====================================================
# TAB 2 - AI Explanation
# ====================================================

with tab2:

    st.header("🧠 AI Explanation")

    if st.session_state.explanation:

        st.write(
            st.session_state.explanation
        )

    else:

        st.info(
            "Run a prediction first."
        )

# ====================================================
# TAB 3 - Medical Chatbot
# ====================================================

with tab3:

    st.header(
        "💬 Medical Assistant Chatbot"
    )

    question = st.text_input(
        "Ask a question about diabetes"
    )

    if st.button(
        "Ask AI",
        key="chat_button"
    ):

        if question.strip():

            with st.spinner(
                "Thinking..."
            ):

                answer = (
                    ask_medical_assistant(
                        question
                    )
                )

            st.success(answer)

        else:

            st.warning(
                "Please enter a question."
            )