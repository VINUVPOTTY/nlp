
🚦 Text Response Classifier
This Streamlit app classifies free-text responses as either flagged (containing sensitive or concerning content) or not_flagged (neutral, positive, or safe).
It is designed to help identify messages that may indicate emotional distress, depression, substance abuse, or other mental health concerns.

How it works
The app uses a machine learning model trained on real-world examples.
User input is preprocessed and vectorized, then classified using a Decision Tree model.
Results are displayed instantly, with clear explanations for each label.
Labels
flagged: Contains sensitive or concerning content (e.g., emotional distress, depression, substance abuse).
not_flagged: Appears neutral, positive, or safe with no concerns.
Usage
Enter any text response in the input box.
Click "Classify Text".
The app will display whether the response is flagged or not.
Tech Stack
Python, Streamlit
scikit-learn, nltk, joblib
Files
app.py — Streamlit app source code
tfidf_vectorizer.joblib — Saved vectorizer
DecisionTreeClassifier_model.joblib — Saved model
requirements.txt — Python dependencie
