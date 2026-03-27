"""
Script to regenerate tokenizer.pkl from the training data.
This tokenizer is needed by the backend to convert text into sequences.
"""
import pandas as pd
import pickle
import os

# Try TensorFlow Keras first, fall back to standalone keras
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
except ImportError:
    from keras.preprocessing.text import Tokenizer

# Load the training data
print("Loading training data...")
df = pd.read_csv("data/clean_data.csv")
print(f"Loaded {len(df)} rows")

# Create and fit the tokenizer on the text column
print("Fitting tokenizer...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"].astype(str).values)
print(f"Tokenizer fitted with {len(tokenizer.word_index)} unique words")

# Save the tokenizer
output_path = os.path.join("backend", "model", "tokenizer.pkl")
with open(output_path, "wb") as f:
    pickle.dump(tokenizer, f)

print(f"✅ Tokenizer saved to: {output_path}")
print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
