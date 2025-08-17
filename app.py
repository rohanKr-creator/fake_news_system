import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Setup stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Cleaning function (same as training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r'[^a-z\s]', "", text)                 # keep only letters
    return " ".join([w for w in text.split() if w not in stop_words])

# Streamlit UI
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter news title or content:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.error("❌ This looks like Fake News!")
        else:
            st.success("✅ This looks like Real News!")
