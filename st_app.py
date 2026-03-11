import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinGuard AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    background-color: #080c12 !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* App header */
.app-header {
    background: linear-gradient(135deg, #0d1117 0%, #111827 100%);
    border-bottom: 1px solid #21262d;
    padding: 28px 32px 20px;
    margin-bottom: 24px;
}
.app-header h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
    letter-spacing: -0.5px;
    margin: 0 0 4px;
}
.app-header p {
    color: #555e6e !important;
    font-size: 13px !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 1px;
    margin: 0;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1.5px dashed #21262d !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #00c8ff !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    color: #fff !important;
    padding: 14px 24px !important;
    width: 100% !important;
    box-shadow: 0 0 20px rgba(14,165,233,0.25) !important;
    transition: box-shadow 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 32px rgba(14,165,233,0.45) !important;
    transform: translateY(-1px) !important;
}

/* Dark panels */
.result-panel, .info-panel {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
}

/* Section headers */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: #555e6e;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Disclaimer */
.disclaimer {
    background: #0d1117;
    border-top: 1px solid #21262d;
    padding: 14px 28px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #3d444d;
    letter-spacing: 0.5px;
    text-align: center;
    margin-top: 24px;
}

/* Streamlit image captions */
[data-testid="caption"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    color: #555e6e !important;
    letter-spacing: 1px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model Architecture ────────────────────────────────────────────────────────
def get_trained_architecture(num_classes=7):
    model = models.resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

LESION_CLASSES = [
    "Benign Keratosis", "Melanocytic Nevus", "Dermatofibroma",
    "Melanoma (Malignant)", "Vascular Lesion", "Basal Cell Carcinoma (Malignant)",
    "Actinic Keratosis / IEC (Precancerous)"
]

MODEL_PATH = "resnet34.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"⚠ {MODEL_PATH} not found — using random weights for demo.")
        return get_trained_architecture(len(LESION_CLASSES)).to(device).eval()

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = None

    if isinstance(checkpoint, dict):
        if "state_dicts" in checkpoint:
            inner = checkpoint["state_dicts"]
            state_dict = next(iter(inner.values())) if isinstance(inner, dict) else inner[0]
        elif "fold_1" in checkpoint:
            state_dict = checkpoint["fold_1"]
        elif any("fc.4.weight" in k or "fc.weight" in k for k in checkpoint.keys()):
            state_dict = checkpoint

    if state_dict is None:
        raise RuntimeError("Could not find a valid state_dict in the checkpoint file.")

    final_layer_key = "fc.4.weight" if "fc.4.weight" in state_dict else "fc.weight"
    detected_classes = state_dict[final_layer_key].shape[0] if final_layer_key in state_dict else len(LESION_CLASSES)

    model = get_trained_architecture(num_classes=detected_classes)
    clean_sd = {k.replace('module.', '').replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd)
    return model.to(device).eval()

model = load_model()

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Helpers ───────────────────────────────────────────────────────────────────
def create_prob_chart(probs_dict: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    classes = list(probs_dict.keys())
    values  = list(probs_dict.values())
    colors  = [
        '#ff4d4d' if any(r in c for r in ["Malignant", "Carcinoma", "Actinic"]) else '#00c8ff'
        for c in classes
    ]

    bars = ax.barh(classes, values, color=colors, height=0.55, edgecolor='none')
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val*100:.1f}%', va='center', color='#aaaaaa', fontsize=9,
                fontfamily='monospace')

    ax.set_xlim(0, 1.15)
    ax.tick_params(colors='#555e6e', labelsize=9)
    ax.spines[:].set_visible(False)
    ax.xaxis.set_visible(False)
    for label in ax.get_yticklabels():
        label.set_color('#c9d1d9')
        label.set_fontfamily('monospace')
        label.set_fontsize(9)
    ax.grid(axis='x', color='#21262d', linewidth=0.5, linestyle='--')
    plt.tight_layout(pad=1.5)
    return fig


def run_gradcam(model, input_tensor, pil_img, top_idx):
    """Run Grad-CAM; returns visualization array (H×W×3, uint8)."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        target_layers = [model.layer4[-1]]
        targets       = [ClassifierOutputTarget(top_idx)]
        rgb_float     = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            vis = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        return vis
    except Exception as e:
        st.warning(f"Grad-CAM unavailable: {e}")
        return np.array(pil_img.resize((224, 224)))


def analyze(image_array):
    pil_img      = Image.fromarray(image_array).convert('RGB')
    input_tensor = test_transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output        = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

    top_idx      = int(np.argmax(probabilities))
    conf         = probabilities[top_idx]
    label        = LESION_CLASSES[top_idx] if top_idx < len(LESION_CLASSES) else f"Class {top_idx}"
    is_malignant = any(r in label for r in ["Malignant", "Carcinoma", "Actinic"])

    agreement = max(min(1.0 - (np.std(probabilities) / 0.5), 1.0), 0.0)
    cam_img   = run_gradcam(model, input_tensor, pil_img, top_idx)
    probs_dict = {
        LESION_CLASSES[i]: float(probabilities[i])
        for i in range(min(len(probabilities), len(LESION_CLASSES)))
    }

    return label, conf, is_malignant, agreement, cam_img, probs_dict


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>⬡ SKINGUARD AI</h1>
  <p>DERMOSCOPIC LESION ANALYSIS · RESNET-34 · GRAD-CAM ENABLED</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

# ── Left panel ────────────────────────────────────────────────────────────────
with left_col:
    uploaded = st.file_uploader(
        "DERMOSCOPIC IMAGE",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        label_visibility="visible",
    )

    preview_placeholder = st.empty()

    if uploaded:
        pil_preview = Image.open(uploaded).convert("RGB")
        preview_placeholder.image(pil_preview, use_container_width=True, caption="INPUT IMAGE")

    run_btn = st.button("▶  RUN ANALYSIS", disabled=(uploaded is None))

    st.markdown("""
    <div class="info-panel" style="margin-top:12px;font-family:'Space Mono',monospace;
         font-size:10px;color:#3d444d;line-height:1.9;">
      <span style="color:#555e6e;">ℹ MODEL INFO</span><br>
      Architecture · ResNet-34<br>
      Classes &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· 7 lesion types<br>
      Input &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· 224 × 224 px<br>
      XAI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Gradient-CAM
    </div>
    """, unsafe_allow_html=True)

# ── Right panel ───────────────────────────────────────────────────────────────
with right_col:
    results_placeholder = st.empty()
    cam_col, rank_col   = st.columns(2, gap="medium")
    cam_placeholder     = cam_col.empty()
    rank_placeholder    = rank_col.empty()

# ── Default empty state ───────────────────────────────────────────────────────
if not uploaded or not run_btn:
    results_placeholder.markdown("""
    <div style="font-family:'Space Mono',monospace;color:#3d444d;padding:48px;
         text-align:center;font-size:12px;letter-spacing:1px;
         background:#0d1117;border:1px solid #21262d;border-radius:10px;">
      AWAITING IMAGE INPUT…
    </div>
    """, unsafe_allow_html=True)

# ── Run analysis ──────────────────────────────────────────────────────────────
if uploaded and run_btn:
    img_array = np.array(Image.open(uploaded).convert("RGB"))

    with st.spinner("Analysing lesion…"):
        label, conf, is_malignant, agreement, cam_img, probs_dict = analyze(img_array)

    risk_color  = "#ff4d4d" if is_malignant else "#00e676"
    risk_icon   = "🔴" if is_malignant else "🟢"
    risk_label  = "HIGH RISK" if is_malignant else "LOW RISK"
    risk_detail = ("High-risk indicators detected — clinical review advised."
                   if is_malignant else "Typical benign features observed.")

    results_placeholder.markdown(f"""
    <div class="result-panel" style="font-family:'Courier New',monospace;">
      <div style="color:#555e6e;font-size:11px;letter-spacing:2px;margin-bottom:6px;">PRIMARY DIAGNOSIS</div>
      <div style="color:#e6edf3;font-size:22px;font-weight:700;margin-bottom:4px;">{label}</div>
      <div style="display:flex;gap:32px;margin-top:12px;">
        <div>
          <div style="color:#555e6e;font-size:10px;letter-spacing:1.5px;">CONFIDENCE</div>
          <div style="color:#00c8ff;font-size:28px;font-weight:700;">
            {conf*100:.1f}<span style="font-size:14px;color:#555">%</span>
          </div>
        </div>
        <div>
          <div style="color:#555e6e;font-size:10px;letter-spacing:1.5px;">MODEL AGREEMENT</div>
          <div style="color:#a8b2c1;font-size:28px;font-weight:700;">
            {agreement*100:.1f}<span style="font-size:14px;color:#555">%</span>
          </div>
        </div>
      </div>
      <div style="background:#1a0a0a;border-left:4px solid {risk_color};padding:10px 16px;
                  border-radius:4px;margin-top:16px;">
        <span style="color:{risk_color};font-weight:700;font-size:13px;">{risk_icon} {risk_label}</span>
        <span style="color:#888;font-size:12px;margin-left:12px;">{risk_detail}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Grad-CAM
    cam_placeholder.image(cam_img, caption="GRAD-CAM ACTIVATION MAP", use_container_width=True)

    # Top-3 confidence ranking
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    rank_md = '<div class="result-panel"><div class="section-label">TOP-3 CONFIDENCE RANKING</div>'
    for i, (cls, prob) in enumerate(sorted_probs, 1):
        bar_color = '#ff4d4d' if any(r in cls for r in ["Malignant", "Carcinoma", "Actinic"]) else '#00c8ff'
        rank_md += f"""
        <div style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-family:'Space Mono',monospace;font-size:11px;color:#c9d1d9;">
              {i}. {cls}
            </span>
            <span style="font-family:'Space Mono',monospace;font-size:11px;color:{bar_color};">
              {prob*100:.1f}%
            </span>
          </div>
          <div style="background:#21262d;border-radius:4px;height:6px;">
            <div style="background:{bar_color};width:{prob*100:.1f}%;height:6px;border-radius:4px;"></div>
          </div>
        </div>"""
    rank_md += '</div>'
    rank_placeholder.markdown(rank_md, unsafe_allow_html=True)

    # Full probability chart
    st.markdown('<div class="section-label" style="margin-top:8px;">FULL PROBABILITY DISTRIBUTION</div>',
                unsafe_allow_html=True)
    fig = create_prob_chart(probs_dict)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  FOR RESEARCH AND EDUCATIONAL USE ONLY · NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL DIAGNOSIS ·
  ALWAYS CONSULT A QUALIFIED DERMATOLOGIST
</div>
""", unsafe_allow_html=True)