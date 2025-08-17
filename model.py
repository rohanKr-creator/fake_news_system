import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 1. Setup NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 2. Dataset folder
dataset_path = "dataset/News _dataset"

# Automatically detect True/Fake CSV files (case-insensitive)
files = os.listdir(dataset_path)
real_file = next(f for f in files if "true" in f.lower())
fake_file = next(f for f in files if "fake" in f.lower())

# Load datasets
real = pd.read_csv(os.path.join(dataset_path, real_file))
fake = pd.read_csv(os.path.join(dataset_path, fake_file))
print("✅ Loaded files:", real_file, fake_file)

# 3. Label data
real['label'] = 0  # Real news
fake['label'] = 1  # Fake news

# Combine datasets
df = pd.concat([real, fake], ignore_index=True)

# 4. Combine title + text into one column
df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
df = df[['content', 'label']].dropna()

# 5. Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r'[^a-z\s]', "", text)                 # keep only letters
    return " ".join([w for w in text.split() if w not in stop_words])

df['clean_text'] = df['content'].apply(clean_text)

# 6. Features & labels
X = df['clean_text']
y = df['label']

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9. Evaluate
y_pred = model.predict(X_test)
print("✅ Training complete")
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# 10. Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("💾 model.pkl and vectorizer.pkl saved successfully!")
