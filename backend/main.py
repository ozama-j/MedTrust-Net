"""
MedTrust-Net FastAPI backend .

"""
import os
import io
import base64
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


try:
    import skimage.morphology as morph
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from model import (
    MedTrustNet, PlainResNet50, GradCAM,
    PATHOLOGIES, NUM_CLASSES, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
    get_preprocess_transform,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="MedTrust-Net API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CHECKPOINT_DIR = Path(os.environ.get('CHECKPOINT_DIR', './checkpoints'))
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant')
MC_DROPOUT_SAMPLES = 20

medtrust_model = None
baseline_model = None
gradcam = None
preprocess = get_preprocess_transform()


def load_models():
    global medtrust_model, baseline_model, gradcam

    medtrust_model = MedTrustNet(num_classes=NUM_CLASSES, pretrained=False).to(device)
    medtrust_ckpt_path = CHECKPOINT_DIR / 'best.pt'
    if medtrust_ckpt_path.exists():
        ckpt = torch.load(medtrust_ckpt_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        medtrust_model.load_state_dict(state)
        print(f"Loaded MedTrust-Net from {medtrust_ckpt_path}")
    else:
        print(f"WARNING: MedTrust-Net checkpoint not found at {medtrust_ckpt_path}")
    medtrust_model.eval()

    baseline_model = PlainResNet50(num_classes=NUM_CLASSES, pretrained=False).to(device)
    baseline_ckpt_path = CHECKPOINT_DIR / 'resnet50_baseline.pt'
    if baseline_ckpt_path.exists():
        state = torch.load(baseline_ckpt_path, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        baseline_model.load_state_dict(state)
        print(f"Loaded baseline from {baseline_ckpt_path}")
    else:
        print(f"WARNING: Baseline checkpoint not found at {baseline_ckpt_path}")
    baseline_model.eval()

    gradcam = GradCAM(baseline_model, baseline_model.layer4)


@app.on_event("startup")
def startup():
    load_models()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def overlay_heatmap(image_np, heatmap, cmap_name='jet', alpha=0.5):
    cmap = cm.get_cmap(cmap_name)
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_rgb = cmap(heatmap_norm)[..., :3]
    overlay = (1 - alpha) * image_np + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(overlay); ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def image_to_base64(image_np):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image_np); ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def denormalize(tensor):
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(arr, 0, 1)


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, img_tensor, n_samples=MC_DROPOUT_SAMPLES):
    model.eval()
    enable_dropout(model)
    sample_probs = []
    for _ in range(n_samples):
        logits = model(img_tensor)
        sample_probs.append(torch.sigmoid(logits).cpu().numpy())
    sample_probs = np.array(sample_probs)
    mean_probs = sample_probs.mean(axis=0)[0]
    std_probs = sample_probs.std(axis=0)[0]
    model.eval()
    return mean_probs, std_probs


def analyze_dam_crm_overlap(dam, crm):
    """
    Analyze spatial relationship between DAM and CRM.
    
    Uses raw CRM values (not normalized) so the uncertainty 
    reading is meaningful and varies across images.
    """

    dam_norm = (dam - dam.min()) / (dam.max() - dam.min() + 1e-8)

    crm_raw = crm  # shape H×W, values typically in [0.3, 1.5]

    # High-attention mask: top 40% of DAM (more selective than median)
    dam_threshold = np.percentile(dam_norm, 60)
    high_attn_mask = dam_norm >= dam_threshold

    unc_in_attended = float(crm_raw[high_attn_mask].mean()) \
        if high_attn_mask.any() else 0.0
    unc_in_ignored  = float(crm_raw[~high_attn_mask].mean()) \
        if (~high_attn_mask).any() else 0.0

    # Thresholds based on actual σ magnitude from the CAB
    # σ < 0.7  → model's attention distributions are tight → confident
    # σ 0.7–1.0 → moderate spread
    # σ > 1.0  → distributions are wide → genuinely uncertain
    if unc_in_attended < 0.70:
        interpretation = (
            "The model shows high confidence in the regions it is attending to. "
            "The CRM indicates tight attention distributions in diagnostically "
            "relevant areas this prediction is well-supported by the model's "
            "internal reasoning."
        )
    elif unc_in_attended < 1.00:
        interpretation = (
            "The model shows moderate uncertainty in the regions it is attending to. "
            "The CRM indicates some spread in the attention distributions across "
            "diagnostically relevant areas — independent review is advisable."
        )
    else:
        interpretation = (
            "The model shows high uncertainty in the very regions it is attending to. "
            "The CRM indicates wide attention distributions where the DAM highlights — "
            "this prediction should be independently verified before clinical use."
        )

    return {
        'uncertainty_in_attended_regions': round(unc_in_attended, 3),
        'uncertainty_in_ignored_regions':  round(unc_in_ignored, 3),
        'interpretation': interpretation,
    }


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "medtrust_loaded": medtrust_model is not None,
        "baseline_loaded": baseline_model is not None,
        "pathologies": PATHOLOGIES,
        "version": "3.0",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if medtrust_model is None or baseline_model is None:
        raise HTTPException(503, "Models not loaded")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    img_t = preprocess(img).unsqueeze(0).to(device)

    # MedTrust-Net
    with torch.no_grad():
        mt_logits, dam, crm = medtrust_model(img_t, return_maps=True)
        mt_probs = torch.sigmoid(mt_logits)[0].cpu().numpy()

    dam_up = F.interpolate(dam, size=(IMAGE_SIZE, IMAGE_SIZE),
                           mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    crm_up = F.interpolate(crm, size=(IMAGE_SIZE, IMAGE_SIZE),
                           mode='bilinear', align_corners=False)[0, 0].cpu().numpy()

    # Spatial uncertainty analysis 
    spatial_analysis = analyze_dam_crm_overlap(dam_up, crm_up)

    # Baseline + Grad-CAM
    with torch.no_grad():
        b_logits = baseline_model(img_t)
        b_probs = torch.sigmoid(b_logits)[0].cpu().numpy()
    top_class_baseline = int(np.argmax(b_probs))
    cam = gradcam(img_t.clone(), top_class_baseline)

    # MC Dropout
    mcd_mean_probs, mcd_std_probs = mc_dropout_predict(baseline_model, img_t)
    top_class_mcd = int(np.argmax(mcd_mean_probs))
    mcd_mean_uncertainty = float(mcd_std_probs.mean())

    img_disp = denormalize(img_t[0])

    return {
        "filename": file.filename,
        "pathologies": PATHOLOGIES,
        "medtrust_net": {
            "probabilities": dict(zip(PATHOLOGIES, mt_probs.tolist())),
            "top_prediction": PATHOLOGIES[int(np.argmax(mt_probs))],
            "top_probability": float(max(mt_probs)),
            # Raw CRM stats 
            "crm_mean_sigma": float(crm_up.mean()),
            "crm_max_sigma": float(crm_up.max()),
            # The actual clinically meaningful analysis
            "spatial_analysis": spatial_analysis,
        },
        "baseline_resnet50": {
            "probabilities": dict(zip(PATHOLOGIES, b_probs.tolist())),
            "top_prediction": PATHOLOGIES[top_class_baseline],
            "top_probability": float(max(b_probs)),
        },
        "mc_dropout": {
            "probabilities": dict(zip(PATHOLOGIES, mcd_mean_probs.tolist())),
            "uncertainty_per_class": dict(zip(PATHOLOGIES, mcd_std_probs.tolist())),
            "top_prediction": PATHOLOGIES[top_class_mcd],
            "top_probability": float(mcd_mean_probs[top_class_mcd]),
            "mean_uncertainty": mcd_mean_uncertainty,
            "n_samples": MC_DROPOUT_SAMPLES,
        },
        "images": {
            "original": image_to_base64(img_disp),
            "medtrust_dam": overlay_heatmap(img_disp, dam_up, cmap_name='jet'),
            "medtrust_crm": overlay_heatmap(img_disp, crm_up, cmap_name='hot'),
            "baseline_gradcam": overlay_heatmap(img_disp, cam, cmap_name='jet'),
        },
    }


# ---- Groq explanation ----
class ExplainRequest(BaseModel):
    medtrust_top: str
    medtrust_prob: float
    spatial_interpretation: str
    unc_in_attended: float
    baseline_top: str
    baseline_prob: float
    mcd_top: str
    mcd_prob: float
    mcd_sigma: float
    all_probabilities: dict


@app.post("/explain")
def explain(req: ExplainRequest):
    if not GROQ_API_KEY:
        return {"explanation": "Groq API key not configured. Set GROQ_API_KEY environment variable."}

    try:
        from groq import Groq
    except ImportError:
        return {"explanation": "groq package not installed."}

    all_agree = req.medtrust_top == req.baseline_top == req.mcd_top
    some_agree = (req.medtrust_top == req.baseline_top) or (req.medtrust_top == req.mcd_top)

    prompt = f"""You are a medical AI assistant interpreting chest X-ray results for a clinician. Be factual, concise (4-5 sentences), and never provide a diagnosis or treatment advice.

Three models analyzed the same chest X-ray:

1. MedTrust-Net (the proposed confidence-aware model):
   - Top finding: {req.medtrust_top} (probability: {req.medtrust_prob:.0%})
   - Spatial uncertainty analysis: {req.spatial_interpretation}
   - Uncertainty in attended regions: {req.unc_in_attended:.2f} (0=confident, 1=uncertain)
   - All probabilities: {json.dumps({k: f"{v:.0%}" for k, v in req.all_probabilities.items()})}

2. Baseline ResNet-50 + Grad-CAM (standard XAI, no uncertainty):
   - Top finding: {req.baseline_top} ({req.baseline_prob:.0%})
   - NOTE: Grad-CAM provides a heatmap of important regions but gives NO indication whether the model is confident or guessing — it always looks confident.

3. MC Dropout (global uncertainty only, {MC_DROPOUT_SAMPLES} passes):
   - Top finding: {req.mcd_top} ({req.mcd_prob:.0%})
   - Global σ across classes: {req.mcd_sigma:.3f}
   - NOTE: MC Dropout can only say the model is globally uncertain. It CANNOT localize where in the image the uncertainty comes from. MedTrust-Net's CRM provides this spatial localization.

Models {"all agree" if all_agree else "partially agree" if some_agree else "disagree"} on the top finding.

Explain to a clinician in 4-5 sentences:
1. What the top finding means clinically.
2. What the Confidence Reliability Map (CRM) reveals — specifically whether uncertainty is concentrated in the diagnostically relevant regions or elsewhere.
3. How MedTrust-Net's spatial uncertainty compares to the baseline's lack of uncertainty information and MC Dropout's global-only uncertainty.
4. Whether this case warrants independent review based on the spatial uncertainty pattern.

End with: "This is AI-assisted decision support and does not constitute a medical diagnosis."
"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return {"explanation": resp.choices[0].message.content}
    except Exception as e:
        return {"explanation": f"Groq API error: {str(e)}"}


# ---- PDF Report ----
class ReportRequest(BaseModel):
    filename: str
    medtrust_net: dict
    baseline_resnet50: dict
    mc_dropout: dict
    images: dict
    explanation: Optional[str] = None


@app.post("/report")
def report(req: ReportRequest):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor, grey
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
                                         Table, TableStyle)
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        raise HTTPException(500, "reportlab not installed. Run: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=15*mm, bottomMargin=15*mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
                                  fontSize=18, textColor=HexColor('#1e3a8a'),
                                  alignment=TA_CENTER, spaceAfter=6)
    subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'],
                                     fontSize=10, textColor=grey,
                                     alignment=TA_CENTER, spaceAfter=12)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                               fontSize=13, textColor=HexColor('#1e3a8a'),
                               spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, leading=13)
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'],
                                       fontSize=8, textColor=HexColor('#991b1b'),
                                       leading=11, spaceBefore=12)

    elements = []
    elements.append(Paragraph("MedTrust-Net Diagnostic Report", title_style))
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    elements.append(Paragraph(
        f"Generated: {timestamp} &nbsp;&nbsp;|&nbsp;&nbsp; Source: {req.filename}", subtitle_style))

    mt = req.medtrust_net
    bl = req.baseline_resnet50
    mcd = req.mc_dropout
    sa = mt.get('spatial_analysis', {})

    elements.append(Paragraph("Diagnostic Predictions", h2_style))
    summary_data = [
        ['Model', 'Top Finding', 'Probability', 'Uncertainty Info'],
        ['MedTrust-Net', mt['top_prediction'],
         f"{mt['top_probability']*100:.1f}%",
         f"Spatial: unc. in attended={sa.get('uncertainty_in_attended_regions','N/A')}"],
        ['ResNet-50 + Grad-CAM', bl['top_prediction'],
         f"{bl['top_probability']*100:.1f}%", 'No uncertainty estimate'],
        ['MC Dropout', mcd['top_prediction'],
         f"{mcd['top_probability']*100:.1f}%",
         f"Global σ: {mcd['mean_uncertainty']:.3f} ({mcd['n_samples']} passes)"],
    ]
    summary_table = Table(summary_data, colWidths=[45*mm, 38*mm, 25*mm, 72*mm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0,0), (-1,0), HexColor('#ffffff')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, grey),
        ('BACKGROUND', (0,1), (-1,1), HexColor('#dbeafe')),
        ('ROWBACKGROUNDS', (0,2), (-1,-1), [HexColor('#ffffff'), HexColor('#f3f4f6')]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(summary_table)

    # Spatial analysis
    interp = sa.get('interpretation', '')
    if interp:
        elements.append(Paragraph("Spatial Uncertainty Analysis", h2_style))
        elements.append(Paragraph(interp, body_style))

    elements.append(Paragraph("Per-Pathology Probabilities", h2_style))
    prob_data = [['Pathology', 'MedTrust-Net', 'Baseline', 'MC Dropout', 'MCD σ']]
    for p in PATHOLOGIES:
        prob_data.append([
            p,
            f"{mt['probabilities'][p]*100:.1f}%",
            f"{bl['probabilities'][p]*100:.1f}%",
            f"{mcd['probabilities'][p]*100:.1f}%",
            f"{mcd['uncertainty_per_class'][p]:.3f}",
        ])
    prob_table = Table(prob_data, colWidths=[50*mm, 30*mm, 30*mm, 30*mm, 25*mm])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0,0), (-1,0), HexColor('#ffffff')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor('#ffffff'), HexColor('#f3f4f6')]),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
    ]))
    elements.append(prob_table)

    elements.append(Paragraph("Visualizations", h2_style))
    elements.append(Paragraph(
        "<b>Reading the maps:</b> DAM and Grad-CAM use <i>jet</i> colormap "
        "(red = high attention, blue = low). CRM uses <i>hot</i> colormap "
        "(bright = uncertain, dark = confident). "
        "A trustworthy prediction shows a CRM that is mostly dark where the DAM is bright.",
        body_style))
    elements.append(Spacer(1, 6))

    def b64_to_image(b64_str, width=75*mm):
        img_bytes = base64.b64decode(b64_str)
        return RLImage(io.BytesIO(img_bytes), width=width, height=width)

    img_grid = Table([
        [[Paragraph("<b>Original</b>", body_style), b64_to_image(req.images['original'])],
         [Paragraph("<b>Grad-CAM</b> (no uncertainty)", body_style), b64_to_image(req.images['baseline_gradcam'])]],
        [[Paragraph("<b>MedTrust-Net DAM</b>", body_style), b64_to_image(req.images['medtrust_dam'])],
         [Paragraph("<b>MedTrust-Net CRM</b>", body_style), b64_to_image(req.images['medtrust_crm'])]],
    ], colWidths=[85*mm, 85*mm])
    img_grid.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(img_grid)

    if req.explanation:
        elements.append(Paragraph("AI-Generated Explanation", h2_style))
        elements.append(Paragraph(req.explanation.replace('\n', '<br/>'), body_style))

    elements.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by MedTrust-Net, an AI decision support "
        "system, and does <b>NOT</b> constitute a medical diagnosis. All findings must be "
        "reviewed by a qualified radiologist. The system detects 5 CheXpert pathologies "
        "(Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion) and cannot "
        "identify conditions outside this scope. The Confidence Reliability Map shows spatial "
        "uncertainty in the model's attention and it is not a calibrated probability of diagnostic "
        "correctness.",
        disclaimer_style))

    doc.build(elements)
    buf.seek(0)
    safe_name = req.filename.rsplit('.', 1)[0].replace(' ', '_')
    filename = f"medtrust_report_{safe_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(buf, media_type='application/pdf',
                             headers={'Content-Disposition': f'attachment; filename="{filename}"'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get('PORT', 8000)))