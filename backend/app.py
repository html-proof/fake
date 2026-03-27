from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
import pickle
import gc
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

# ===============================
# MEMORY OPTIMIZATION FOR RAILWAY
# ===============================
# Limit TensorFlow memory growth
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warning logs
tf.config.set_soft_device_placement(True)

# Limit TensorFlow to use minimal threads (saves ~50-100MB)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Keras 3 unified imports
try:
    import keras
    from keras.models import load_model
    from keras.preprocessing.sequence import pad_sequences
    print(f"Using Keras {keras.__version__}")
except (ImportError, AttributeError):
    # Fallback to tf.keras if independent keras isn't there
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    print("Using tensorflow.keras fallback")

import PyPDF2
from docx import Document
from PIL import Image
import tempfile
import pytesseract
import logging

# Configure logging to see errors in the console/logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WINDOWS_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
elif os.name == "nt" and os.path.exists(WINDOWS_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT_PATH
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running", "message": "Fake Job Detection System Ready"}), 200

# ===============================
# FILE UPLOAD CONFIGURATION
# ===============================
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'csv'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "fake_job_bilstm.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")

# ===============================
# LAZY-LOAD MODEL & TOKENIZER
# ===============================
# Use lazy loading to avoid duplicating memory in Gunicorn fork.
_model = None
_tokenizer = None

MAX_LEN = 300

def get_model():
    """Lazy-load the Keras model on first request."""
    global _model
    if _model is None:
        try:
            _model = load_model(MODEL_PATH)
            logger.info("Model loaded successfully (lazy)")
        except Exception as e:
            logger.error(f"FAILED TO LOAD MODEL: {str(e)}")
    return _model

def get_tokenizer():
    """Lazy-load the tokenizer on first request."""
    global _tokenizer
    if _tokenizer is None:
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)
            if not hasattr(_tokenizer, 'num_words') or _tokenizer.num_words is None or _tokenizer.num_words > 50000:
                _tokenizer.num_words = 50000
        logger.info("Tokenizer loaded successfully (lazy)")
    return _tokenizer

print("Model and tokenizer will be loaded on first request")

# ===============================
# LIGHTWEIGHT EXPLANATION (replaces LIME)
# ===============================
def get_keyword_explanation(text):
    """Return suspicious keywords found in the text as explanation.
    This replaces LIME which ran 5000+ predictions per call and caused OOM."""
    suspicious_keywords = [
        "no interview", "no documents", "apply fast", "limited slots",
        "work from home", "no experience", "urgent hiring", "immediate joining",
        "high salary", "contact us today", "whatsapp", "telegram",
        "guaranteed income", "easy money", "part time data entry",
        "earning potential", "good salary", "dm me", "inbox me",
        "fast-growing startup", "impact millions", "fictional", "demo only"
    ]
    text_lower = text.lower()
    found = [kw for kw in suspicious_keywords if kw in text_lower]
    return found if found else ["No obvious red flags detected"]

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
    elif file_ext == 'csv':
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            # Combine all text/object columns into a single blob for analysis
            text_cols = df.select_dtypes(include=['object']).columns
            return " ".join(df[text_cols].astype(str).values.flatten())
        except Exception:
            # Simple fallback using standard csv module
            import csv
            lines = []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for row in reader:
                    lines.append(" ".join(row))
            return " ".join(lines)
    
    return None

# ===============================
# STRONG RULE-BASED SCAM CHECK
# ===============================
def rule_based_scam_check(text):
    brand_names = [
        "google", "amazon", "microsoft", "infosys", "tcs", "wipro", "hcl", "accenture"
    ]

    # Blocklist of known scams/fakes
    fake_companies = [
        "nexaplay", "nexaplay technologies", "career-nexaplay",
        "jobguard-demo", "fake-inc"
    ]

    off_platform_contact = [
        "whatsapp", "telegram", "contact hr", "dm me", "message me on", "inbox me"
    ]

    # Suspicious phrases including those found in unrealistic startup postings
    scam_phrases = [
        "no interview", "no documents", "apply fast", "limited slots",
        "part time data entry", "work from home", "no experience",
        "urgent hiring", "immediate joining", "high salary",
        "contact us today", "good salary", "earning potential",
        "next-generation", "impact millions", "fast-growing startup",
        "fictional", "practice project", "demo only",
        "limited bandwidth", "emerging markets"
    ]

    hits = 0
    text_lower = text.lower()

    # 🔴 Strong scam: Brand impersonation + Off-platform contact
    if any(b in text_lower for b in brand_names) and any(c in text_lower for c in off_platform_contact):
        hits += 4

    # 🔴 Known Fake Company / Domain Found (Strong Indicator)
    if any(fake in text_lower for fake in fake_companies):
        hits += 3

    # 🔴 Potential fake domain / email pattern
    if ".io" in text_lower or ".xyz" in text_lower or ".me" in text_lower:
        # These domains aren't inherently scams, but common in "demo" or "startup" fakes
        hits += 1

    # 🟡 Suspicious phrases
    for phrase in scam_phrases:
        if phrase in text_lower:
            hits += 1

    # 🟡 Unusually high salary for "no experience"
    if "no experience" in text_lower and ("salary" in text_lower or "lpa" in text_lower):
        hits += 2

    return hits

# ===============================
# PREDICTION ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    raw_text = data.get("text", "")

    try:
        if not raw_text or raw_text.strip() == "":
            return jsonify({"error": "No text provided"}), 400

        text = preprocess_text(raw_text)

        model = get_model()
        tokenizer = get_tokenizer()

        sequence = tokenizer.texts_to_sequences([text])
        # Safety filter: ensure all word indices are within embedding layer range [0, 50000)
        sequence = [[min(idx, 49999) for idx in seq] for seq in sequence]
        
        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        # ML Prediction
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        preds = model.predict(padded, verbose=0)
        prob_fake = float(preds[0][0])

        # Lightweight keyword explanation (replaces LIME to avoid OOM)
        explanation = get_keyword_explanation(raw_text)

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

        # Force garbage collection to free memory after prediction
        gc.collect()

        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 2),
            "incomplete_description": incomplete_flag,
            "explanation": explanation
        })
    except Exception as e:
        gc.collect()  # Clean up even on error
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

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
        model = get_model()
        tokenizer = get_tokenizer()

        sequence = tokenizer.texts_to_sequences([text])

        # SAFETY CLIP: Ensure indices don't exceed model embedding limit (50,000)
        clipped_sequence = [[min(idx, 49999) for idx in sequence[0]]]
        
        padded = pad_sequences(
            clipped_sequence,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
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

        # Force garbage collection to free memory
        gc.collect()
        
        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 2),
            "incomplete_description": incomplete_flag,
            "file_name": secure_filename(file.filename),
            "extracted_text_length": len(extracted_text)
        })
    
    except Exception as e:
        gc.collect()
        logger.error(f"File prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

# ===============================
# FEEDBACK COLLECTION (LEARNING MODE)
# ===============================
@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    """Save user-reported misclassifications for future training."""
    data = request.get_json()
    raw_text = data.get("text", "")
    correct_label = data.get("correct_label", "")
    reported_at = data.get("timestamp", "")

    if not raw_text or not correct_label:
        return jsonify({"error": "No data provided"}), 400

    try:
        feedback_file = os.path.join(BASE_DIR, "model", "training_feedback.json")
        feedback_data = []

        # Load existing feedback
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                try:
                    feedback_data = json.load(f)
                except json.JSONDecodeError:
                    feedback_data = []

        # Add new feedback
        feedback_data.append({
            "text": raw_text,
            "correct_label": correct_label,
            "timestamp": reported_at,
            "status": "unverified"
        })

        # Save back to file
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=4)

        logger.info(f"Feedback received for training data: {correct_label}")
        return jsonify({"success": "Feedback saved for future training"}), 200
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000"))
    )
