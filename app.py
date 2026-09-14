import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

# Page Configuration
st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="⚙️", layout="wide")

st.title("⚙️ AI-Powered Industrial Predictive Maintenance Dashboard")
st.write("An end-to-end Machine Learning web application designed to predict industrial machine failures and analyze sensor feature importance.")

@st.cache_resource
def load_train_and_evaluate_model():
    try:
        df = pd.read_csv('ai4i2020.csv')
    except FileNotFoundError:
        st.error("Dataset 'ai4i2020.csv' not found. Please upload it to your GitHub repository.")
        return None, None, None, None, None, None, None, None

    # Clean data & drop extra columns
    cols_to_drop = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    features_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    X = df.drop(columns=features_to_drop)
    y = df['Machine failure']
    
    # Feature names for later usage
    feature_names = list(X.columns)
    
    # Encode categorical column 'Type'
    if 'Type' in X.columns:
        le = LabelEncoder()
        X['Type'] = le.fit_transform(X['Type'])
    else:
        le = None
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluation metrics
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, scaler, le, feature_names, X_test_scaled, y_test, acc, roc_auc, conf_matrix

# Load model data
result = load_train_and_evaluate_model()
if result[0] is not None:
    model, scaler, le, feature_names, X_test_scaled, y_test, acc, roc_auc, conf_matrix = result

    # Create Tabs for Clean Navigation
    tab1, tab2 = st.tabs(["🔍 Live Predictor", "📊 Model Analytics & Insights"])

    with tab1:
        st.subheader("Test Real-Time Machine Sensor Readings")
        st.write("Adjust the slider parameters in the sidebar to simulate machine conditions and predict failure risk.")

        st.sidebar.header("🔧 Sensor Inputs")
        type_val = st.sidebar.selectbox("Machine Type (0: L, 1: M, 2: H)", [0, 1, 2])
        air_temp = st.sidebar.slider("Air Temperature [K]", 295.0, 305.0, 298.0)
        process_temp = st.sidebar.slider("Process Temperature [K]", 305.0, 315.0, 308.0)
        rot_speed = st.sidebar.slider("Rotational Speed [rpm]", 1100, 2900, 1500)
        torque = st.sidebar.slider("Torque [Nm]", 3.0, 80.0, 40.0)
        tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 250, 100)

        if st.button("Predict Machine Status", type="primary"):
            input_data = np.array([[type_val, air_temp, process_temp, rot_speed, torque, tool_wear]])
            input_scaled = scaler.transform(input_data)
            
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.error(f"⚠️ **High Risk of Failure!**")
                else:
                    st.success(f"✅ **Normal Operation**")
            with col2:
                st.metric(label="Calculated Failure Probability", value=f"{probability:.2%}")

    with tab2:
        st.subheader("Model Performance Metrics")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Model Accuracy", value=f"{acc:.4f}")
        with col_m2:
            st.metric(label="ROC-AUC Score", value=f"{roc_auc:.4f}")
            
        st.markdown("---")
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.write("### Feature Importance")
            st.write("Which sensors contribute most to predicting failures?")
            importances = model.feature_importances_
            fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
            sns.barplot(x=importances, y=feature_names, ax=ax_fi, palette="Blues_r")
            ax_fi.set_title("Random Forest Feature Importances")
            st.pyplot(fig_fi)
            
        with col_v2:
            st.write("### Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            ax_cm.set_title("Test Set Confusion Matrix")
            st.pyplot(fig_cm)
