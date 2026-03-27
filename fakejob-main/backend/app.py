from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import numpy as np
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
import pytesseract
from PIL import Image
import tempfile
import pytesseract

WINDOWS_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.exists(WINDOWS_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT_PATH
app = Flask(__name__)
CORS(app)

# ===============================
# FILE UPLOAD CONFIGURATION
# ===============================
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "fake_job_bilstm.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")

# ===============================
# LOAD MODEL & TOKENIZER
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("🔥 Model and tokenizer loaded successfully 🔥")

MAX_LEN = 300
explainer = LimeTextExplainer(class_names=["Real Job", "Fake Job"])
def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    preds = model.predict(padded)
    return np.hstack((1 - preds, preds))

# ===============================
# TEXT PREPROCESSING
# ===============================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ===============================
# INCOMPLETE JOB CHECK
# ===============================
def is_incomplete_job(text):
    keywords = [
        "job title",
        "company",
        "location",
        "job description",
        "responsibilities",
        "requirements"
    ]
    count = sum(1 for kw in keywords if kw in text)
    return count < 2

# ===============================
# FILE EXTRACTION HELPERS
# ===============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    """Extract text from PDF file using PyPDF2."""
    try:
        text = ""
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text if text.strip() else None
    except Exception as e:
        return None

def extract_text_from_docx(filepath):
    """Extract text from DOCX file using python-docx."""
    try:
        doc = Document(filepath)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text if text.strip() else None
    except Exception as e:
        return None

def extract_text_from_image(filepath):
    """Extract text from image using pytesseract OCR."""
    try:
        image = Image.open(filepath)
        text = pytesseract.image_to_string(image)
        return text if text.strip() else None
    except Exception as e:
        return None

def extract_text_from_txt(filepath):
    """Extract text from TXT file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        return text if text.strip() else None
    except Exception as e:
        return None

def extract_text_from_file(filepath, file_ext):
    """Route to appropriate text extraction function based on file type."""
    file_ext = file_ext.lower()
    
    if file_ext == 'pdf':
        return extract_text_from_pdf(filepath)
    elif file_ext == 'docx':
        return extract_text_from_docx(filepath)
    elif file_ext in ['png', 'jpg', 'jpeg']:
        return extract_text_from_image(filepath)
    elif file_ext == 'txt':
        return extract_text_from_txt(filepath)
    
    return None

# ===============================
# STRONG RULE-BASED SCAM CHECK
# ===============================
def rule_based_scam_check(text):
    brand_names = [
        "google", "amazon", "microsoft", "infosys", "tcs", "wipro"
    ]

    off_platform_contact = [
        "whatsapp", "telegram", "contact hr"
    ]

    # UPDATED suspicious patterns
    scam_phrases = [
        "no interview",
        "no documents",
        "apply fast",
        "limited slots",
        "part time data entry",
        "work from home",
        "no experience",
        "urgent hiring",
        "immediate joining",
        "high salary",
        "contact us today"
        "Good Salary"
    ]

    hits = 0

    # 🔴 Strong scam: Brand impersonation + WhatsApp
    if any(b in text for b in brand_names) and any(c in text for c in off_platform_contact):
        hits += 3

    # 🔴 Strong scam: No experience + salary promise
    if "no experience" in text and ("salary" in text or "earn" in text):
        hits += 1

    # 🟡 Suspicious vague phrases
    hits += sum(1 for phrase in scam_phrases if phrase in text)

    return hits

# ===============================
# PREDICTION ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    raw_text = data.get("text", "")

    if not raw_text or raw_text.strip() == "":
        return jsonify({"error": "No text provided"}), 400

    text = preprocess_text(raw_text)

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prob_fake = float(model.predict(padded, verbose=0)[0][0])

    # LIME explanation
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=5
    )

    explanation = [word for word, score in exp.as_list()]

    # ML decision
    if prob_fake >= 0.5:
        result = "Fake Job"
        confidence = prob_fake * 100
    else:
        result = "Real Job"
        confidence = (1 - prob_fake) * 100

    # RULE-BASED CHECK
    rule_hits = rule_based_scam_check(text)

    if rule_hits >= 3:
        result = "Fake Job"
        confidence = 90.0

    elif rule_hits >= 2 and prob_fake < 0.5:
        result = "Suspicious / Likely Fake"
        confidence = 70.0

    incomplete_flag = is_incomplete_job(text)

    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2),
        "incomplete_description": incomplete_flag,
        "explanation": explanation
    })

# ===============================
# FILE UPLOAD PREDICTION ROUTE
# ===============================
@app.route("/predict-file", methods=["POST"])
def predict_file():
    """Handle file upload and predict if job posting is Real/Fake."""
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Accepted formats: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    try:
        # Create temporary directory to save file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, secure_filename(file.filename))
            file.save(filepath)
            
            # Check file size
            file_size = os.path.getsize(filepath)
            if file_size > MAX_FILE_SIZE:
                return jsonify({"error": f"File size exceeds {MAX_FILE_SIZE / 1024 / 1024}MB limit"}), 400
            
            # Extract text based on file type
            file_ext = file.filename.rsplit('.', 1)[1].lower()
            extracted_text = extract_text_from_file(filepath, file_ext)
            
            # Check if text was successfully extracted
            if not extracted_text:
                return jsonify({"error": f"Could not extract readable text from {file_ext.upper()} file"}), 400
        
        # Preprocess extracted text
        text = preprocess_text(extracted_text)
        
        # Check if text is too short
        if len(text.split()) < 10:
            return jsonify({"error": "Extracted text is too short for analysis. Please provide a more complete job posting."}), 400
        
        # Generate prediction using existing pipeline
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )
        
        prob_fake = float(model.predict(padded, verbose=0)[0][0])
        
        # ===============================
        # ML-BASED DECISION (BASE)
        # ===============================
        if prob_fake >= 0.5:
            result = "Fake Job"
            confidence = prob_fake * 100
        else:
            result = "Real Job"
            confidence = (1 - prob_fake) * 100
        
        # ===============================
        # RULE-BASED OVERRIDE
        # ===============================
        rule_hits = rule_based_scam_check(text)
        
        # 🔴 Strong scam override
        if rule_hits >= 3:
            result = "Fake Job"
            confidence = 90.0
        
        # 🟡 Borderline suspicious override
        elif rule_hits >= 2 and prob_fake < 0.5:
            result = "Suspicious / Likely Fake"
            confidence = 70.0
        
        # ===============================
        # INCOMPLETE FLAG
        # ===============================
        incomplete_flag = is_incomplete_job(text)
        
        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 2),
            "incomplete_description": incomplete_flag,
            "file_name": secure_filename(file.filename),
            "extracted_text_length": len(extracted_text)
        })
    
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
