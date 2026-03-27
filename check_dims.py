import tensorflow as tf
from keras.models import load_model
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = r"c:\Users\seban\Downloads\backend-main\backend-main\backend\model\fake_job_bilstm.keras"
TOKENIZER_PATH = r"c:\Users\seban\Downloads\backend-main\backend-main\backend\model\tokenizer.pkl"

print("--- Model Info ---")
model = load_model(MODEL_PATH)
model.summary()

print("\n--- Tokenizer Info ---")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print(f"Tokenizer num_words: {tokenizer.num_words}")
print(f"Tokenizer word_index length: {len(tokenizer.word_index)}")
