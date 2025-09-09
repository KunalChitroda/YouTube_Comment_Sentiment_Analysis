import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

# Load your dataset
vaders = pd.read_csv('youtube_comments_sentiment.csv')

# Split the data
x = vaders['Comments']
y = vaders['Sentiment Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=60)

# Load your pre-trained TF-IDF vectorizer
tfidf = TfidfVectorizer()  # Use your actual vectorizer

# Fit the vectorizer on your training data
x_train_vect = tfidf.fit_transform(x_train)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_vect, y_train)

# Define a Streamlit app
def main():
    st.title("YouTube Comment Sentiment Analysis")
    st.write("Analyzing comments from YouTube videos")

    # Collect user input
    user_comment = st.text_input("Enter a comment:")

    if user_comment:
        # Preprocess user input
        user_input_vect = tfidf.transform([user_comment])

        # Predict the sentiment using the KNN model
        prediction = knn.predict(user_input_vect)

        # Map the sentiment category back to a human-readable label
        sentiment_label = "Positive" if prediction == 1 else "Negative"

        st.subheader("Sentiment Analysis Result:")
        st.write(f"Predicted Sentiment: {sentiment_label}")

    st.sidebar.header("About")
    st.sidebar.info("This app predicts the sentiment of user-provided comments.")
    st.sidebar.header("Instructions")
    st.sidebar.info("Enter a comment to predict its sentiment as positive, negative or Neutral.")
    st.sidebar.header("Dataset Source")
    st.sidebar.markdown("[YouTube Comments Dataset](https://docs.google.com/spreadsheets/d/1vKdmfqGbuSfO4zCMKnFFgfnQpz3mRRn6rTacgu6uOi8/edit#gid=1397941421)")

if __name__ == "__main__":
    main()


# streamlit run youtube_sentiment_analysis.py