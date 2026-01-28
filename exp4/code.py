
from transformers import pipeline

# load pretrained model
sentiment_model = pipeline("sentiment-analysis")

# your dataset
reviews = [
    "very good movie",
    "excellent acting",
    "good story",
    "amazing direction",
    "enjoyable film",
    "bad movie",
    "poor acting",
    "boring story",
    "terrible direction",
    "not enjoyable"
]

print("Pretrained Model Predictions:\n")

for text in reviews:
    result = sentiment_model(text)[0]
    print(f"Review: {text}  -->  {result['label']}  (score={result['score']:.2f})")
