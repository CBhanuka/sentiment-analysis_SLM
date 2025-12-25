from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Simple HTML form to input the review
    return """
    <html>
        <body>
            <h1>Sentiment Analysis</h1>
            <form action="/predict/" method="post">
                <label for="review">Enter a Review:</label><br><br>
                <input type="text" id="review" name="review" required><br><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

@app.post("/predict/")
async def predict(review: str = Form(...)):
    # Preprocess the review and predict sentiment
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    sentiment = 'Good' if prediction == 1 else 'Bad'
    return {"Review": review, "Prediction": sentiment}
