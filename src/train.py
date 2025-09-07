import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent  # src/ -> project root

# Ensure directories exist
(PROJECT_ROOT / 'models').mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(PROJECT_ROOT / 'data' / 'dataset.csv')
texts = df['review'].tolist()
labels = df['sentiment'].map({'positive': 1, 'negative': 0}).tolist()

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Evaluate
predictions = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# Save model and vectorizer
joblib.dump(model, PROJECT_ROOT / 'models' / 'model.joblib')
joblib.dump(vectorizer, PROJECT_ROOT / 'models' / 'vectorizer.joblib')

print("Model trained and saved!")