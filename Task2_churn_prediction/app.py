import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="SentinelAI Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()

except:
    st.error("Model files not found. Run train_model.py first")
    st.stop()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Hide Streamlit top bar */
header {
    visibility: hidden;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
    color: white;
}

/* Main glass container */
.main-box {
    background: rgba(255,255,255,0.06);
    padding: 2rem;
    border-radius: 24px;
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 25px rgba(0,255,255,0.12);
}

/* Titles */
h1, h2, h3 {
    color: white !important;
    font-weight: 800 !important;
}

/* Labels */
label, p {
    color: #e5e7eb !important;
    font-weight: 600 !important;
}

/* Input styling */
.stSelectbox div,
.stNumberInput div,
.stSlider div {
    color: white !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    color: white !important;
    border-radius: 14px;
    border: none;
    padding: 0.8rem;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 18px rgba(59,130,246,0.5);
}

/* Result box */
.result-box {
    padding: 1.5rem;
    border-radius: 18px;
    text-align: center;
    margin-top: 1rem;
    font-size: 28px;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("📌 SentinelAI")

    st.write("AI Powered Customer Churn Prediction")

    st.write("---")

    st.write("### Features")

    st.write("✔ Customer churn prediction")

    st.write("✔ AI business analytics")

    st.write("✔ Risk analysis dashboard")

    st.write("✔ Probability prediction")

    st.write("---")

    st.write("Developer: Rose ❤️🎀")

# ---------------- MAIN UI ----------------
st.markdown('<div class="main-box">', unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction Dashboard")

st.write(
    "Analyze customer behavior and predict churn risk using Machine Learning."
)

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:

    credit_score = st.slider(
        "Credit Score",
        300,
        900,
        650
    )

    age = st.slider(
        "Age",
        18,
        100,
        35
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        5
    )

    balance = st.number_input(
        "Account Balance",
        0.0,
        300000.0,
        50000.0
    )

    salary = st.number_input(
        "Estimated Salary",
        0.0,
        300000.0,
        50000.0
    )

with col2:

    products = st.slider(
        "Number of Products",
        1,
        4,
        1
    )

    has_card = st.selectbox(
        "Has Credit Card",
        ["Yes", "No"]
    )

    active = st.selectbox(
        "Active Member",
        ["Yes", "No"]
    )

    geography = st.selectbox(
        "Country",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

# ---------------- CONVERT VALUES ----------------
gender = 1 if gender == "Male" else 0

active = 1 if active == "Yes" else 0

has_card = 1 if has_card == "Yes" else 0

geo_map = {
    "France": 0,
    "Germany": 1,
    "Spain": 2
}

geography = geo_map[geography]

# ---------------- DATAFRAME ----------------
data = pd.DataFrame([[
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    products,
    has_card,
    active,
    salary
]], columns=[
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
])

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Churn Risk"):

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1] * 100

    st.write("")

    if probability >= 40:

        st.markdown(f"""
        <div class="result-box"
        style="background:#7f1d1d;color:white;">
        🚨 HIGH CHURN RISK
        <br><br>
        Churn Probability: {probability:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "This customer may leave the service soon. "
            "Consider retention offers and loyalty rewards."
        )

    else:

        st.markdown(f"""
        <div class="result-box"
        style="background:#064e3b;color:white;">
        ✅ CUSTOMER LIKELY TO STAY
        <br><br>
        Stay Probability: {100 - probability:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.success(
            "Customer appears stable with low churn probability."
        )

    # Progress bar
    st.markdown("### Churn Probability")

    st.progress(int(probability))

    # Metrics
    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Prediction",
        "CHURN" if probability >= 40 else "STAY"
    )

    m2.metric(
        "Confidence",
        f"{max(probability, 100-probability):.2f}%"
    )

    if probability >= 80:
        risk = "Critical"

    elif probability >= 60:
        risk = "High"

    elif probability >= 40:
        risk = "Medium"

    else:
        risk = "Low"

    m3.metric(
        "Risk Level",
        risk
    )

st.markdown("</div>", unsafe_allow_html=True)