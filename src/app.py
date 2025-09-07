import gradio as gr
import joblib
from pathlib import Path

# Get the project root (one level up from src/)
PROJECT_ROOT = Path(__file__).parent.parent  # src/ -> project root

# Load model and vectorizer
model = joblib.load(PROJECT_ROOT / 'models' / 'model.joblib')
vectorizer = joblib.load(PROJECT_ROOT / 'models' / 'vectorizer.joblib')

# Rest of the code remains the same
def predict_sentiment(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return "Positive" if prediction == 1 else "Negative"

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Enter text for sentiment analysis"),
    outputs=gr.Textbox(label="Sentiment"),
    title="Sentiment Analysis Demo",
    description="Enter a movie review to predict if it's positive or negative."
)

iface.launch()