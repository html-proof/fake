# 🚀 Railway Deployment Guide for Backend

Follow these steps to deploy this project successfully on [Railway.app](https://railway.app).

## 1. Environment Variables
Add these variables in the **Variables** tab of your Railway service:

| Variable | Value | Description |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.11.11` | Ensures the correct Python version is used. |
| `NIXPACKS_PKGS` | `tesseract` | **Critical:** Installs Tesseract OCR for image text extraction. |
| `PORT` | `8080` | (Optional) Railway handles this, but good to have. |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | (Optional) Suppresses TensorFlow startup noise. |

## 2. Deployment Details
- **Build Command**: Railway automatically detects `requirements.txt`.
- **Start Command**: Railway uses the `Procfile` (`web: gunicorn backend.app:app`).
- **Nixpacks**: Railway uses Nixpacks by default, which reads the `NIXPACKS_PKGS` variable to install system dependencies like Tesseract.

## 3. Post-Deployment
Once the build is successful:
1. Go to **Settings** > **Networking**.
2. Click **Generate Domain**.
3. Use this URL for your API requests (e.g., `https://your-app.up.railway.app/predict`).

---
*Note: The project structure has been updated so that `requirements.txt` and `Procfile` are in the root directory.*
