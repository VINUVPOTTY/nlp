import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z' ]", '', text)
    return text

# Load vectorizer and model
vec = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('DecisionTreeClassifier_model.joblib')

def main():
    st.markdown("""
        <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .glow-title {
            text-align: center;
            color: #000011;
            text-shadow: 0 0 5px #FFD700, 0 0 10px #FFA500, 0 0 20px #FFA500, 0 0 30px #FF8C00;
            font-size: 2.8em;
            margin-bottom: 20px;
        }
        .subtitle-bold {
            font-weight: 800;
            font-size: 1.1em;
            color: black;
        }
        .bold-label {
            font-weight: bold;
            font-size: 1.1em;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='glow-title'>🚦 Text Response Classifier</h1>", unsafe_allow_html=True)

    st.markdown("""
        <div class='info-box'>
            <p class='subtitle-bold'>✨ What do the labels mean?</p>
            <ul>
                <li><b>flagged</b>: Contains sensitive or concerning content (e.g., emotional distress, depression, substance abuse).</li>
                <li><b>not_flagged</b>: Appears neutral, positive, or safe with no concerns.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='bold-label'>💬 Enter a response to classify:</p>", unsafe_allow_html=True)
    user_input = st.text_area("", label_visibility='collapsed')

    if st.button("🔍 Classify Text"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a valid response.")
        else:
            cleaned_input = clean_text(user_input)
            input_vec = vec.transform([cleaned_input])
            prediction = model.predict(input_vec)[0]
            label_map = {0: "not_flagged", 1: "flagged"}
            predicted_label = label_map[prediction]
            if predicted_label == "flagged":
                st.error(f"🚨 Prediction: **{predicted_label.upper()}**")
            else:
                st.success(f"✅ Prediction: **{predicted_label.upper()}**")

if __name__ == '__main__':
    main()