from transformers import pipeline
import torch    

# Initialize the sentiment-analysis pipeline once.
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text using the sentiment-analysis pipeline.

    Args:
        text (str): The text to analyze.

    Returns:
        tuple: (label, score) where label is either "POSITIVE" or "NEGATIVE" and score is a float between 0 and 1.
    """
    result = classifier(text)[0]
    return result["label"], result["score"]
