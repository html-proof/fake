# Railway Deployment Guide for Backend

Follow these steps to deploy this project successfully on Railway.

## 1. Variables to add
Because this repo has a root `Dockerfile`, Railway will build with Docker for this service.

Add these in the Railway service Variables tab:

| Variable | Value | Required | Why |
| :--- | :--- | :--- | :--- |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | No | Reduces noisy TensorFlow startup logs. |
| `TESSERACT_CMD` | `tesseract` | No | Only use this if Railway cannot find Tesseract automatically. |

Do not set `PORT` manually. Railway injects `PORT` for you, and the container is now configured to bind to it.

You also do not need `PYTHON_VERSION` here because the Docker image already pins Python.

## 2. Build and start behavior
- Railway will use the root `Dockerfile`.
- The Docker image installs `tesseract-ocr`, `libgl1`, and `libglib2.0-0` during build.
- The deployment is pinned by `railway.toml` to start with `./start.sh`.
- The startup script reads Railway's injected `PORT` and launches Gunicorn safely.

```dockerfile
CMD ["./start.sh"]
```

## 3. Add variables quickly
This repo includes a root `.env.example`, which you can copy into Railway's Variables tab.

## 4. After deploy
1. Open your Railway service.
2. Go to Settings > Networking.
3. Generate a public domain.
4. Test the health route:

```text
https://your-service.up.railway.app/
```

5. Test prediction:

```text
https://your-service.up.railway.app/predict
```

## 5. Notes
- The health endpoint is `/`.
- Large TensorFlow model loading can make cold starts slower than a typical Flask app.
- User feedback submitted to `/submit-feedback` is written to the local container filesystem, so it will not be durable across redeploys unless you later add a database or volume.
