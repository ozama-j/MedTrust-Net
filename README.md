# MedTrust-Net

**A Confidence-Aware Attention Architecture for Unified Explainability and Uncertainty Quantification in Medical Diagnosis**

Final-year project — Jalal Hushama, ID- 2084767, BSc Computer Science, University of Westminster.

---

## What this is

MedTrust-Net is a deep learning architecture that simultaneously produces a diagnostic prediction, an attention map showing *what* the model is looking at, and a **confidence reliability map** showing *how sure* it is about each region. It addresses the trust gap in medical AI where Grad-CAM-style explanations always look confident even when the model is guessing.

The novel contribution is the **Confidence-Aware Attention Block (CAB)** — a stochastic attention layer that learns spatial and channel attention as Gaussian distributions (mean μ + log-variance log σ²) instead of deterministic weights, via the reparameterization trick. The mean becomes the DAM, the variance becomes the CRM, all in a single forward pass.

---

## Getting started (clone & run)

Model checkpoints are **not** stored in this repo (too large for GitHub) — download them from Google Drive first.

### Prerequisites

- Python 3.11+
- Node.js 18+
- A [Groq API key](https://console.groq.com) (for the `/explain` endpoint)

### 1. Clone the repo

```bash
git clone https://github.com/ozama-j/MedTrust-Net.git
cd MedTrust-Net
```

### 2. Download the model checkpoints

Download `best.pt` and `resnet50_baseline.pt` from Google Drive and place them in `backend/checkpoints/`:

- **Google Drive folder:** <
https://drive.google.com/drive/folders/1_2rG7sKxcezrMlvx0RGHupftvajYvvXq?usp=sharing
```bash
mkdir -p backend/checkpoints
# move/copy the downloaded best.pt and resnet50_baseline.pt into backend/checkpoints/
```

### 3. Run the backend

```bash
cd backend
pip install -r requirements.txt

export GROQ_API_KEY="your_groq_key"
export CHECKPOINT_DIR="./checkpoints"

uvicorn main:app --reload --port 8000
```

### 4. Run the frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Open http://localhost:5173 — the backend must be running on port 8000 for predictions to work.

---

## Repository layout

```
medtrust-net/
├── notebooks/
│   ├── 01_medtrust_train.ipynb            # train the hero model (CAB + hybrid loss)
│   └── 02_baselines_eval_ablations.ipynb  # baselines, ablations, all benchmarking visualition
│   
├── backend/
│   ├──checkpoint                          #best.pt and restnet50_baseline.pt
│   ├── main.py                            # FastAPI app
│   ├── model.py                           # CAB + MedTrustNet + PlainResNet50 (importable)
│   ├── requirements.txt
│   ├── Dockerfile                         # for HF Spaces deployment
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── App.jsx                        # main UI
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
├── results/
│   ├── figures/         # PNGs (created by notebooks)
│   ├── metrics/         # JSON results (created by notebooks)
│   └── checkpoints/     # best.pt, baseline.pt etc. (created by notebooks)
└── README.md
```

---

## Retraining from scratch (optional)

Only needed if you want to reproduce the checkpoints yourself instead of using the pretrained ones from Google Drive.

### 1. Train the hero model

1. Place `CheXpert-v1.0-small.zip` in `MyDrive/` on Google Drive
2. Upload `notebooks/01_medtrust_train.ipynb` to Colab
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells. Stage 1 (frozen) takes ~20 min, Stage 2 (unfrozen) takes 1.5–3 hours
5. Best model is saved to `MyDrive/medtrust_results/checkpoints/best.pt`

### 2. Run baselines + ablations

1. After Notebook A finishes, upload `notebooks/02_baselines_eval_ablations.ipynb`
2. Run all cells. Trains 3 ablation variants + 2 ensemble members + evaluates everything
3. Produces all benchmarking figures in `MyDrive/medtrust_results/figures/`

### 3. Run the backend locally

```bash
cd backend
pip install -r requirements.txt

# Copy checkpoints from your Drive
mkdir -p checkpoints
cp /path/to/MyDrive/medtrust_results/checkpoints/best.pt checkpoints/
cp /path/to/MyDrive/medtrust_results/checkpoints/resnet50_baseline.pt checkpoints/

export GROQ_API_KEY="your_groq_key"
export CHECKPOINT_DIR="./checkpoints"

uvicorn main:app --reload --port 8000
```

### 4. Run the frontend locally

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

---

## Targets 

- Macro AUROC ≥ 0.85 ✅
- Macro ECE ≤ 0.10 ✅
- Inference < 100 ms ✅
- Model size < 200 MB ✅

---

## Ablation study (Notebook B)

Full 2×2 grid as required by the supervisor's evaluation criteria:

|  | No Cal Loss | + Cal Loss |
|---|---|---|
| **No CAB** | ResNet-50 baseline | ResNet-50 + Cal |
| **+ CAB**  | CAB only | **MedTrust-Net (full)** |

Plus the three baselines from the PPRS:
- Grad-CAM on ResNet-50
- MC Dropout (20 samples)
- Deep Ensemble (3 ResNet-50s)

---


