import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load and preprocess data
def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Encode the labels (Good=1, Bad=0)
    data['Label'] = data['Label'].map({'Good': 1, 'Bad': 0})

    # Split the data into features (X) and labels (y)
    X = data['Review']
    y = data['Label']

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Train the model
def train_model():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('reviews.csv')

    # Convert text to numerical features using TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

    # Save the trained model and vectorizer
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

if __name__ == "__main__":
    train_model()
