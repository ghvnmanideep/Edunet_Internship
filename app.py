import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('models/LR_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("Supply Chain Emissions Prediction")

st.markdown("""
This app predicts **Supply Chain Emission Factors with Margins** based on DQ metrics and other parameters.
""")

# Input form
with st.form("prediction_form"):
    substance = st.selectbox("Substance", ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
    unit = st.selectbox("Unit", ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'])
    source = st.selectbox("Source", ['Commodity', 'Industry'])
    supply_wo_margin = st.number_input("Supply Chain Emission Factors without Margins", min_value=0.0)
    margin = st.number_input("Margins of Supply Chain Emission Factors", min_value=0.0)
    dq_reliability = st.slider("DQ Reliability", 0.0, 1.0)
    dq_temporal = st.slider("DQ Temporal Correlation", 0.0, 1.0)
    dq_geo = st.slider("DQ Geographical Correlation", 0.0, 1.0)
    dq_tech = st.slider("DQ Technological Correlation", 0.0, 1.0)
    dq_data = st.slider("DQ Data Collection", 0.0, 1.0)

    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        'Substance': substance,
        'Unit': unit,
        'Supply Chain Emission Factors without Margins': supply_wo_margin,
        'Margins of Supply Chain Emission Factors': margin,
        'DQ ReliabilityScore of Factors without Margins': dq_reliability,
        'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
        'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
        'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
        'DQ DataCollection of Factors without Margins': dq_data,
        'Source': source,
    }

    input_df = pd.DataFrame([input_data])
    
    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, columns=['Substance', 'Unit', 'Source'], drop_first=True)

    # Reindex to match the expected feature names
    expected_columns = scaler.feature_names_in_  # Get the feature names used during fitting
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)  # Fill missing columns with 0

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Supply Chain Emission Factor with Margin: **{prediction[0]:.4f}**")
