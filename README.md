# ⚙️ Industrial Predictive Maintenance Engine & Web App

An end-to-end Machine Learning web application designed to forecast industrial equipment failures using operational sensor telemetry. Built to bridge raw industrial data with real-time predictive decision-making.

🔗 **Live App Demo:** [Access Streamlit Dashboard Here](https://predictive-maintenance-model-7koeyfbsumfkopxceekldg.streamlit.app/)

---
📊 Dataset & Features
- The model processes key telemetric indicators and quality variant classifications:

- Machine Type : Low (L), Medium (M) and High (H) quality variants governing physical stress thresholds.

- Temperature Sensors: Air temperature and Process temperature [K].

- Kinematic Metrics: Rotational speed [rpm] and Torque [Nm].

- Operational Wear: Tool wear [min].

- Target: Machine failure (Binary classification).

🛠️ Tech Stack
- Core Language: Python

- Data Processing: NumPy, Pandas

- Machine Learning: Scikit-Learn (RandomForestClassifier, StandardScaler, LabelEncoder)

- Visualizations: Matplotlib, Seaborn

- Deployment & UI: Streamlit Cloud

📈 Key Insights & Model Performance
- Non-Linear Modeling: Leveraged a tuned Random Forest Classifier to capture complex interaction terms between torque, tool wear and machine quality variants.

- Feature Importance: Analysis reveals that Torque and Tool Wear serve as the leading indicators for mechanical overstrain and failure.

- Evaluation Metrics: Validated using precision, recall, ROC-AUC score and custom confusion matrices designed for real-world imbalanced classification tasks.

  ---

💻 How to Run Locally
 1. Clone the repository:
   ```
   git clone https://github.com/vaidya21/Predictive-Maintenance-Model.git
   cd Predictive-Maintenance-Model
   ```
 2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
 3. Launch the web application locally:
   ```
   streamlit run app.py
   ```
   
