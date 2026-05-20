import streamlit as st
import joblib

st.set_page_config(page_title="SentinelAI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
header {visibility: hidden;}
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e1b4b);
}
.main-card {
    background: rgba(17, 24, 39, 0.82);
    padding: 2rem;
    border-radius: 24px;
    border: 1px solid rgba(99, 102, 241, 0.35);
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.15);
}
h1, h2, h3, h4, p, label {
    color: #f8fafc !important;
}
textarea {
    background: #0b1220 !important;
    color: #f8fafc !important;
    font-size: 18px !important;
    border-radius: 12px !important;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white !important;
    font-weight: 600 !important;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
}
pre {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #e5e7eb !important;
    border-radius: 12px !important;
}
code {
    color: #e5e7eb !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

def get_risk_level(spam_prob):
    if spam_prob >= 0.85:
        return "Critical"
    elif spam_prob >= 0.60:
        return "High"
    elif spam_prob >= 0.30:
        return "Medium"
    return "Low"

def find_suspicious_words(text):
    keywords = [
        "free", "win", "winner", "claim", "click", "urgent",
        "prize", "offer", "congratulations", "call now",
        "limited", "cash", "reward"
    ]
    text = text.lower()
    return [word for word in keywords if word in text]

try:
    model, vectorizer = load_model()
except Exception:
    st.error("Model files not found. Run train_model.py first.")
    st.stop()

if "message" not in st.session_state:
    st.session_state["message"] = ""

spam_example = "Congratulations! You've won a free iPhone. Click now!"
ham_example = "Hey, are we meeting at 5 PM today?"

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.title("🛡️ SentinelAI")
st.subheader("Intelligent SMS Threat Detection System")

left, right = st.columns([3, 1])

with left:
    message = st.text_area(
        "Enter an SMS message to analyze",
        value=st.session_state["message"],
        height=180
    )

with right:
    st.markdown("### Quick Examples")

    if st.button("Use Spam Example"):
        st.session_state["message"] = spam_example
        st.rerun()

    if st.button("Use Safe Example"):
        st.session_state["message"] = ham_example
        st.rerun()

    st.code(spam_example)
    st.code(ham_example)

if st.button("Analyze Message"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        X = vectorizer.transform([message])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        ham_prob = probabilities[0]
        spam_prob = probabilities[1]
        confidence = max(probabilities) * 100
        risk_level = get_risk_level(spam_prob)
        suspicious_words = find_suspicious_words(message)

        if prediction == 1:
            st.error("Potential spam message detected.")
        else:
            st.success("Message appears legitimate.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", "SPAM" if prediction == 1 else "HAM")
        c2.metric("Confidence", f"{confidence:.2f}%")
        c3.metric("Risk Level", risk_level)

        st.markdown("### Probability Analysis")
        st.progress(int(spam_prob * 100))
        st.write(f"Spam Probability: {spam_prob * 100:.2f}%")
        st.write(f"Ham Probability: {ham_prob * 100:.2f}%")

        st.markdown("### Suspicious Keywords")
        if suspicious_words:
            st.write(", ".join(suspicious_words))
        else:
            st.write("No suspicious keywords detected.")

        st.markdown("### Security Recommendation")
        if prediction == 1:
            st.warning("Avoid clicking unknown links and do not share OTPs, passwords, or banking details.")
        else:
            st.info("No major threat indicators were found in this message.")

st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Project Overview")
    st.write("Model: TF-IDF + Multinomial Naive Bayes")
    st.write("Accuracy: 96.86%")
    st.write("Internship: CodSoft Machine Learning")
    st.write("Developer: Rose")
