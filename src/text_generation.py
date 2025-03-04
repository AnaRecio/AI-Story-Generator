from huggingface_hub import InferenceClient
import torch

def generate_text(user_query, client, max_tokens=100):
    """
    Generates text using the Gemma model via the Hugging Face Inference API.

    Args:
        user_query (str): The prompt to generate text from.
        max_tokens (int): The maximum number of tokens to generate.
        client (InferenceClient): The initialized Hugging Face client.

    Returns:
        str: The generated text from the model.
    """
    message = [{"role": "user", "content": user_query}]
    completion = client.chat.completions.create(
        model="google/gemma-2-27b-it",
        messages=message,
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=0.5
    )

    # Return the generated text
    return completion.choices[0].message.content
