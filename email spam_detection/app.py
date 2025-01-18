import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Function to load or train the spam detection model
def load_or_train_model():
    try:
        # Attempt to load the pre-trained model
        with open('spam_detection_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        return model, vectorizer
    except FileNotFoundError:
        # If model file does not exist, train a new model
        return train_spam_model()

# Train the spam model and save it
def train_spam_model():
    # Load the dataset (replace with your dataset)
    data = pd.read_csv('spam.csv', encoding='latin-1')

    # Preprocessing the data
    data = data[['Message', 'Category']]
    data['spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

    # Splitting the data into train and test sets
    X = data['Message']
    y = data['spam']
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Text vectorization using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    train_X_vec = vectorizer.fit_transform(train_X)
    test_X_vec = vectorizer.transform(test_X)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(train_X_vec, train_Y)

    # Predict and evaluate the model
    y_pred = model.predict(test_X_vec)
    accuracy = accuracy_score(test_Y, y_pred)
    print(f"Model Accuracy: {accuracy*100:.2f}%")

    # Save the model and vectorizer
    with open('spam_detection_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

    return model, vectorizer

# Streamlit app
def main():
    st.title("Email Spam Detection")

    # Load or train the model
    model, vectorizer = load_or_train_model()

    # Create a text input for the user to enter an email
    email_text = st.text_area("Enter the email content", "")

    if st.button("Check Spam"):
        if email_text:
            # Vectorize the input text
            email_vec = vectorizer.transform([email_text])

            # Predict the class (spam or non-spam)
            prediction = model.predict(email_vec)

            if prediction == 1:
                st.write("This email is classified as: **Not Spam**")
            else:
                st.write("This email is classified as: **Spam**")
        else:
            st.write("Please enter some email content to classify.")

# Run the app
if __name__ == '__main__':
    main()
