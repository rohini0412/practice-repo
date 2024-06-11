import streamlit as st
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the saved model and vectorizer
model = joblib.load('lgbm_multi_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define the labels
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Streamlit application
st.title("Toxic Comment Classification")

# Input text from the user
comment_text = st.text_area("Enter a comment:")

# If the user clicks the button
if st.button("Classify"):
    if comment_text:
        # Preprocess the input text
        preprocessed_text = preprocess_text(comment_text)

        # Transform the input text using the loaded TfidfVectorizer
        X_tfidf = vectorizer.transform([preprocessed_text])

        # Predict using the trained model
        y_pred_proba = model.predict_proba(X_tfidf)
        y_pred = model.predict(X_tfidf)

        # Check if the comment is toxic
        is_toxic = np.any([y_pred_proba[i][0][1] > 0.5 for i in range(len(label_columns))])

        if is_toxic:
            st.write("The comment is **toxic**.")
            st.write(y_pred_proba)
            st.write(y_pred)
            st.write("Toxicity intensity and categories:")

            # Display the probability and category
            for i, label in enumerate(label_columns):
                st.write(f"{label}: {y_pred_proba[i][0][1]:.2f}")

        else:
            st.write("The comment is **non-toxic**.")
            st.write(y_pred_proba)
            st.write(y_pred)
            
    else:
        st.write("Please enter a comment to classify.")
