---
title: MedTrust-Net API
emoji: 🩺
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# MedTrust-Net Backend

FastAPI service that wraps the trained MedTrust-Net + ResNet-50 baseline for the React frontend.

## Local run

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
export CHECKPOINT_DIR="/path/to/checkpoints"
uvicorn main:app --reload --port 8000
```

## Required checkpoint files in `CHECKPOINT_DIR`

- `best.pt` — MedTrust-Net (from Notebook A)
- `resnet50_baseline.pt` — Plain baseline (from Notebook B)

## Endpoints

- `GET /health`
- `POST /predict` — file upload, returns predictions + base64 PNG overlays
- `POST /explain` — JSON body, returns Groq-generated explanation
