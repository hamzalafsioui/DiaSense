import os

import mlflow.sklearn
import pandas as pd
import streamlit as st

from diasense.config import (
    CLINICAL_FEATURES,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

# Page configuration
st.set_page_config(
    page_title="DiaSense | Diabetes Risk Prediction",
    page_icon=None,
    layout="centered",
)

# Model loading
# Load once and cache so every user interaction does not reload from MLflow.
@st.cache_resource
def load_model():
    """Load the Production pipeline from the MLflow Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)



# Risk result display
def show_result(prediction: int):
    

    is_high_risk = prediction == 1

    # --- Risk badge ---
    if is_high_risk:
        st.markdown(
            """
            <div style="
                background-color: #c0392b;
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                font-size: 22px;
                font-weight: bold;
                letter-spacing: 1px;
            ">
                HIGH RISK --> Diabetes Risk Detected
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                background-color: #27ae60;
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                font-size: 22px;
                font-weight: bold;
                letter-spacing: 1px;
            ">
                LOW RISK --> No Diabetes Risk Detected
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")  # spacing

    # --- Follow-up advice ---
    st.subheader("Follow-up Advice")

    if is_high_risk:
        st.warning(
            "This patient falls into the high-risk group. "
            "The following actions are recommended:"
        )
        st.markdown(
            """
            - Schedule a fasting blood glucose test within 2 weeks.
            - ...
            - ...
            - ...
            """
        )
    else:
        st.info(
            "This patient falls into the low-risk group. "
            "Standard preventive measures are recommended:"
        )
        st.markdown(
            """
            - Encourage a balanced diet and regular physical activity.
            - ...
            - ...
            - ...
            """
        )


# Main page
def main():
    st.title("DiaSense --> Diabetes Risk Prediction")
    st.markdown(
        "Enter the patient's clinical values below and press **Predict** "
        "to estimate their diabetes risk group."
    )
    st.divider()

    # --- Load model ---
    try:
        model = load_model()
    except Exception as exc:
        st.error(
            f"Could not load the model from MLflow.\n\n"
            f"Make sure the MLflow server is running at `{MLFLOW_TRACKING_URI}` "
            f"and the model `{MLFLOW_REGISTERED_MODEL_NAME}` is registered.\n\n"
            f"Error: {exc}"
        )
        st.stop()

    # --- Patient input form ---
    st.subheader("Patient Clinical Data")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0, max_value=20, value=1, step=1,
                help="Number of times the patient has been pregnant.",
            )
            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=0.0, max_value=300.0, value=120.0, step=1.0,
                help="Plasma glucose concentration (2-hour oral glucose tolerance test).",
            )
            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0.0, max_value=200.0, value=70.0, step=1.0,
                help="Diastolic blood pressure.",
            )
            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0.0, max_value=100.0, value=20.0, step=1.0,
                help="Triceps skinfold thickness.",
            )

        with col2:
            insulin = st.number_input(
                "Insulin (mu U/mL)",
                min_value=0.0, max_value=900.0, value=80.0, step=1.0,
                help="2-hour serum insulin level.",
            )
            bmi = st.number_input(
                "BMI (kg/m^2)",
                min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                help="Body mass index.",
            )
            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0, max_value=3.0, value=0.3, step=0.01,
                help="A function that scores the likelihood of diabetes based on family history.",
            )
            age = st.number_input(
                "Age (years)",
                min_value=1, max_value=120, value=30, step=1,
                help="Patient age in years.",
            )

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    # --- Prediction ---
    if submitted:
        # Build the input DataFrame in the exact same column order as training
        patient_data = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure, skin_thickness,
              insulin, bmi, dpf, age]],
            columns=CLINICAL_FEATURES,
        )

        # Cast to float so the pipeline always receives the expected dtype
        patient_data = patient_data.astype(float)

        prediction = model.predict(patient_data)[0]

        st.divider()
        st.subheader("Prediction Result")
        show_result(int(prediction))


if __name__ == "__main__":
    main()
