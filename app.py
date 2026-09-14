import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Page Configuration
st.set_page_config(page_title="Predictive Maintenance App", page_icon="⚙️", layout="centered")

st.title("⚙️ Industrial Predictive Maintenance App")
st.write("This app uses a Random Forest Machine Learning model to predict machine failure based on telemetric sensor data.")

@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv('ai4i2020.csv')
    except FileNotFoundError:
        st.error("Dataset 'ai4i2020.csv' not found in the repository. Please upload it to GitHub.")
        return None, None, None

    # Drop identifiers and target columns, including the specific failure mode flags if present
    cols_to_drop = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    features_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    X = df.drop(columns=features_to_drop)
    y = df['Machine failure']
    
    # Encode categorical column 'Type'
    if 'Type' in X.columns:
        le = LabelEncoder()
        X['Type'] = le.fit_transform(X['Type'])
    else:
        le = None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, le
    
    return model, scaler, le

model, scaler, le = load_and_train_model()

if model is not None:
    st.sidebar.header("🔧 Input Sensor Metrics")
    
    # Inputs corresponding to AI4I dataset features
    type_val = st.sidebar.selectbox("Machine Type (0: L, 1: M, 2: H)", [0, 1, 2])
    air_temp = st.sidebar.slider("Air Temperature [K]", 295.0, 305.0, 298.0)
    process_temp = st.sidebar.slider("Process Temperature [K]", 305.0, 315.0, 308.0)
    rot_speed = st.sidebar.slider("Rotational Speed [rpm]", 1100, 2900, 1500)
    torque = st.sidebar.slider("Torque [Nm]", 3.0, 80.0, 40.0)
    tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 250, 100)

    # Prediction Button
    if st.button("Predict Machine Status", type="primary"):
        input_data = np.array([[type_val, air_temp, process_temp, rot_speed, torque, tool_wear]])
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        st.subheader("Prediction Results:")
        if prediction == 1:
            st.error(f"⚠️ **Warning: High Risk of Machine Failure!** (Failure Probability: {probability:.2%})")
        else:
            st.success(f"✅ **Machine is Operating Normally.** (Failure Probability: {probability:.2%})")
