import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="ChurnSense",
    page_icon="📊",
    layout="wide"
)


# Custom CSS Styling

st.markdown("""
<style>
header {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e1b4b);
}

.main-card {
    background: rgba(17, 24, 39, 0.88);
    padding: 2rem;
    border-radius: 24px;
    border: 1px solid rgba(99, 102, 241, 0.35);
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.15);
}

h1, h2, h3, h4, p, label {
    color: #f8fafc !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white !important;
    font-weight: 700 !important;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.2rem;
    font-size: 16px;
}

[data-testid="stMetricValue"] {
    color: #60a5fa !important;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}
</style>
""", unsafe_allow_html=True)



# Load model files

@st.cache_resource
def load_assets():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, encoders, feature_columns


# Encode categorical values

def encode_value(column, value, encoders):
    if column in encoders:
        encoder = encoders[column]
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        return 0
    return value


# Risk level helper

def get_risk_level(probability):
    if probability >= 0.80:
        return "Critical"
    elif probability >= 0.60:
        return "High"
    elif probability >= 0.30:
        return "Medium"
    else:
        return "Low"


# Load files safely

try:
    model, scaler, encoders, feature_columns = load_assets()
except Exception:
    st.error("Model files not found. Run: python train_model.py")
    st.stop()



# Main UI

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.title("📊 ChurnSense")
st.subheader("Intelligent Banking Customer Churn Prediction System")

st.markdown("### Customer Information")

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

with col2:
    tenure = st.slider(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        max_value=300000.0,
        value=50000.0
    )

    num_products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

with col3:
    has_card = st.selectbox(
        "Has Credit Card",
        [0, 1]
    )

    active_member = st.selectbox(
        "Is Active Member",
        [0, 1]
    )

    salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        max_value=300000.0,
        value=75000.0
    )



# Build input row

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": active_member,
    "EstimatedSalary": salary
}

row = []

for column in feature_columns:
    value = input_data.get(column, 0)

    if isinstance(value, str):
        value = encode_value(column, value, encoders)

    row.append(value)

input_df = pd.DataFrame([row], columns=feature_columns)



# Prediction

if st.button("Predict Churn"):
    scaled_input = scaler.transform(input_df)

    probabilities = model.predict_proba(scaled_input)[0]
    stay_probability = probabilities[0]
    churn_probability = probabilities[1]

    # Business threshold (40%)
    prediction = 1 if churn_probability >= 0.40 else 0

    confidence = max(probabilities) * 100
    risk_level = get_risk_level(churn_probability)

    if prediction == 1:
        st.error("🚨 Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    # Metrics
    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Prediction",
        "CHURN" if prediction == 1 else "STAY"
    )

    m2.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    m3.metric(
        "Risk Level",
        risk_level
    )

    # Probability Analysis
    st.markdown("### Probability Analysis")

    st.progress(int(churn_probability * 100))

    st.write(
        f"📉 Churn Probability: {churn_probability * 100:.2f}%"
    )

    st.write(
        f"📈 Stay Probability: {stay_probability * 100:.2f}%"
    )

    # Recommendations
    st.markdown("### Retention Recommendation")

    if prediction == 1:
        st.warning(
            "Offer loyalty rewards, personalized support, "
            "or special retention benefits to reduce churn risk."
        )
    else:
        st.info(
            "This customer appears stable with a low probability of churn."
        )

st.markdown("</div>", unsafe_allow_html=True)



# Sidebar

with st.sidebar:
    st.header("Project Overview")
    st.write("Model: Random Forest Classifier")
    st.write("Dataset: Bank Customer Churn Dataset")
    st.write("Internship: CodSoft Machine Learning")
    st.write("Developer: Rose ")