# Updated SafeURL Predictor Streamlit App (Model + Scaler, No Mock Data)

import streamlit as st
import re
import pandas as pd
from datetime import date
import time 
import joblib
import warnings
import requests
from urllib.parse import urlparse
import os
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
#            FEATURE EXTRACTION
# -------------------------------------------------------------

def extract_features(url):
    """Extracts lexical features from a URL."""
    try:
        parsed = urlparse(url)
        path = parsed.path
    except:
        path = url

    features = {
        'url_length': len(url),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special_chars': len(re.findall(r"[^A-Za-z0-9]", url)),
        'has_https': 1 if url.lower().startswith("https") else 0,
        'num_subdomains': url.count('.'),
        'contains_login': 1 if "login" in url.lower() else 0,
        'contains_pay': 1 if "pay" in url.lower() else 0,
    }
    return pd.DataFrame([features])

# -------------------------------------------------------------
#            MODEL + SCALER LOADING (CACHED)
# -------------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    """Load model from HuggingFace and scaler locally from the repo."""

    model_url = "https://huggingface.co/Nayds004/url_prediction_model/resolve/main/ensemble_model.joblib"
    
    # Local scaler path (same directory as the app)
    scaler_path = "scaler.joblib"

    # Download model if not already cached locally
    if not os.path.exists("ensemble_model.joblib"):
        model_bytes = requests.get(model_url).content
        with open("ensemble_model.joblib", "wb") as f:
            f.write(model_bytes)

    # Load model
    model = joblib.load("ensemble_model.joblib")

    # Load scaler locally (NO requests.get!)
    scaler = joblib.load(scaler_path)

    return model, scaler


# -------------------------------------------------------------
#            URL VALIDATION
# -------------------------------------------------------------

def is_valid_url(url):
    url_regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https:// or ftp://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ipv4
        r'(?::\d+)?'  # port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(url_regex, url) is not None

# -------------------------------------------------------------
#            STREAMLIT PAGE CONFIG
# -------------------------------------------------------------

def set_page_config():
    st.set_page_config(
        page_title="SafeURL Predictor",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# -------------------------------------------------------------
#            INFO PAGE
# -------------------------------------------------------------

def info_page():
    st.title("🛡️ SafeURL Predictor: Project Overview")
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1e90ff;">
        <h2 style="color: #1e90ff;">Project Mission</h2>
        <p>
        This application is designed to analyze and classify URLs in real-time to determine if they are safe or malicious.
        It uses a machine learning model trained on lexical URL features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🎓 Student Information")
    st.info("**Name:** Chinedu Egbuna")
    st.info("**Course:** Data Science Project")
    st.info(f"**Date:** {date.today().strftime('%B %d, %Y')}")

# -------------------------------------------------------------
#            PREDICTION PAGE
# -------------------------------------------------------------

def prediction_page():
    st.title("🔗 URL Safety Checker")
    st.subheader("Enter a URL to check if it’s safe.")

    st.markdown("---")

    url_input = st.text_input(
        "URL to Analyze",
        value="https://www.google.com/search?q=safeurl",
        help="Paste a full URL"
    )

    if st.button("Analyze URL", type="primary"):

        if not is_valid_url(url_input):
            st.error("❌ Invalid URL format. Please enter a complete URL starting with http:// or https://.")
            return

        with st.spinner("Extracting features and running ML model..."):
            time.sleep(1)

            features_df = extract_features(url_input)
            # loading the model and scaler files
            model, scaler = load_model_and_scaler()
            scaled = scaler.transform(features_df)
            prediction = model.predict(scaled)[0]

        st.toast("Analysis complete!")
        st.markdown("### Result")

        if prediction == 1:
            st.error("🚨 **MALICIOUS URL**")
        else:
            st.success("✅ **BENIGN URL**")

        st.markdown("### Extracted Features")
        st.json(features_df.iloc[0].to_dict())

# -------------------------------------------------------------
#            MAIN APP
# -------------------------------------------------------------

def main():

    if 'page' not in st.session_state:
        st.session_state.page = "Project Info"

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to:", ("Project Info", "URL Prediction"))
        st.session_state.page = page
        st.markdown("---")
        st.caption(f"App Version 1.0 | {date.today().year}")

    if st.session_state.page == "Project Info":
        info_page()
    else:
        prediction_page()

# -------------------------------------------------------------
#            RUN APP
# -------------------------------------------------------------

if __name__ == "__main__":
    set_page_config()
    main()
