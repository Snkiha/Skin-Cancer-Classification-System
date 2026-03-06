import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# --Model Architecture--
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

# --Load Model and Configurations--
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "resnet34.pth"
LESION_CLASSES = [
    "Benign Keratosis", "Melanocytic Nevus", "Dermatofibroma",
    "Melanoma (Malignant)", "Vascular Lesion", "Basal Cell Carcinoma (Malignant)",
    "Actinic Keratosis / IEC (Precancerous)"
]

def load_full_model():
    if not os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} not found. Loading architecture with random weights.")
        return get_trained_architecture(len(LESION_CLASSES)).to(device).eval()

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = None

    if isinstance(checkpoint, dict):
        if "state_dicts" in checkpoint:
            inner = checkpoint["state_dicts"]
            if isinstance(inner, dict):
                state_dict = next(iter(inner.values()))
            elif isinstance(inner, list):
                state_dict = inner[0]
        elif "fold_1" in checkpoint:
            state_dict = checkpoint["fold_1"]
        elif any("fc.4.weight" in k or "fc.weight" in k for k in checkpoint.keys()):
            state_dict = checkpoint

    if state_dict is None:
        raise RuntimeError("Could not find a valid state_dict in the checkpoint file.")

    final_layer_key = "fc.4.weight" if "fc.4.weight" in state_dict else "fc.weight"
    detected_classes = state_dict[final_layer_key].shape[0] if final_layer_key in state_dict else len(LESION_CLASSES)

    model = get_trained_architecture(num_classes=detected_classes)
    clean_state_dict = {k.replace('module.', '').replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    return model.to(device).eval()

model = load_full_model()

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --Plot Confidence Graph (dark clinical style)--
def create_plot(probs_dict):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    classes = list(probs_dict.keys())
    values = list(probs_dict.values())
    colors = ['#ff4d4d' if any(r in c for r in ["Malignant", "Carcinoma", "Actinic"]) else '#00c8ff' for c in classes]

    bars = ax.barh(classes, values, color=colors, height=0.55, edgecolor='none')

    # Add subtle value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val*100:.1f}%', va='center', color='#aaaaaa', fontsize=9, fontfamily='monospace')

    ax.set_xlabel("Confidence", color='#555e6e', fontsize=9, fontfamily='monospace')
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

# --Prediction Engine--
def analyze_lesion(image):
    if image is None:
        return None, None, None, None

    pil_img = Image.fromarray(image).convert('RGB')
    input_tensor = test_transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probabilities))
    conf = probabilities[top_idx]
    label = LESION_CLASSES[top_idx] if top_idx < len(LESION_CLASSES) else f"Class {top_idx}"
    is_malignant = any(risk in label for risk in ["Malignant", "Carcinoma", "Actinic"])

    # Grad-CAM
    target_layers = [model.layer4[-1]]
    targets = [ClassifierOutputTarget(top_idx)]
    rgb_img_float = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0

    try:
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        visualization = image

    agreement_score = max(min(1.0 - (np.std(probabilities) / 0.5), 1.0), 0.0)

    risk_icon   = "🔴" if is_malignant else "🟢"
    risk_label  = "HIGH RISK" if is_malignant else "LOW RISK"
    risk_color  = "#ff4d4d" if is_malignant else "#00e676"
    risk_banner = (
        f'<div style="background:#1a0a0a;border-left:4px solid {risk_color};padding:10px 16px;'
        f'border-radius:4px;margin-top:10px;font-family:monospace;">'
        f'<span style="color:{risk_color};font-weight:700;font-size:13px;">{risk_icon} {risk_label}</span>'
        f'<span style="color:#888;font-size:12px;margin-left:12px;">'
        + ("High-risk indicators detected — clinical review advised." if is_malignant
           else "Typical benign features observed.")
        + '</span></div>'
    )

    res_md = f"""
<div style="font-family:'Courier New',monospace;background:#0d1117;padding:20px;border-radius:8px;border:1px solid #21262d;">
  <div style="color:#555e6e;font-size:11px;letter-spacing:2px;margin-bottom:6px;">PRIMARY DIAGNOSIS</div>
  <div style="color:#e6edf3;font-size:22px;font-weight:700;margin-bottom:4px;">{label}</div>
  <div style="display:flex;gap:24px;margin-top:12px;">
    <div>
      <div style="color:#555e6e;font-size:10px;letter-spacing:1.5px;">CONFIDENCE</div>
      <div style="color:#00c8ff;font-size:26px;font-weight:700;">{conf*100:.1f}<span style="font-size:14px;color:#555">%</span></div>
    </div>
    <div>
      <div style="color:#555e6e;font-size:10px;letter-spacing:1.5px;">MODEL AGREEMENT</div>
      <div style="color:#a8b2c1;font-size:26px;font-weight:700;">{agreement_score*100:.1f}<span style="font-size:14px;color:#555">%</span></div>
    </div>
  </div>
  {risk_banner}
</div>
"""

    probs_dict = {
        LESION_CLASSES[i]: float(probabilities[i])
        for i in range(min(len(probabilities), len(LESION_CLASSES)))
    }
    plot = create_plot(probs_dict)

    return res_md, probs_dict, plot, visualization


# CSS Configuration
css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

/* Global reset */
body, .gradio-container {
    background: #080c12 !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Remove default Gradio card shadows */
.gr-box, .gr-form { box-shadow: none !important; }

/* Header */
#app-header {
    background: linear-gradient(135deg, #0d1117 0%, #111827 100%);
    border-bottom: 1px solid #21262d;
    padding: 28px 32px 20px;
    margin-bottom: 0;
}
#app-header h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
    letter-spacing: -0.5px;
    margin: 0 0 4px;
}
#app-header p {
    color: #555e6e !important;
    font-size: 13px !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 1px;
    margin: 0;
}

/* Section labels */
label, .label-wrap span {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    color: #555e6e !important;
    text-transform: uppercase !important;
}

/* Upload zone */
#upload-col .gr-image-upload {
    border: 1.5px dashed #21262d !important;
    border-radius: 10px !important;
    background: #0d1117 !important;
    transition: border-color 0.2s;
}
#upload-col .gr-image-upload:hover {
    border-color: #00c8ff !important;
}

/* Analyse button */
#run-btn {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    color: #fff !important;
    padding: 14px !important;
    margin-top: 10px !important;
    width: 100% !important;
    box-shadow: 0 0 20px rgba(14,165,233,0.25) !important;
    transition: box-shadow 0.2s, transform 0.1s !important;
}
#run-btn:hover {
    box-shadow: 0 0 32px rgba(14,165,233,0.45) !important;
    transform: translateY(-1px) !important;
}
#run-btn:active { transform: translateY(0) !important; }

/* Result panels */
.gr-markdown, .gr-plot, .gr-label {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

/* Cam image */
#cam-panel img {
    border-radius: 8px !important;
    border: 1px solid #21262d !important;
}

/* Dividers */
.gr-row { gap: 16px !important; }
.gr-column { gap: 16px !important; }

/* Disclaimer footer */
#disclaimer {
    background: #0d1117;
    border-top: 1px solid #21262d;
    padding: 14px 28px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #3d444d;
    letter-spacing: 0.5px;
    text-align: center;
    margin-top: 4px;
}
"""

# UI Layout 
with gr.Blocks(css=css, title="SkinGuard AI") as demo:

    # Header
    with gr.Group(elem_id="app-header"):
        gr.HTML("""
        <h1>⬡ SKINGUARD AI</h1>
        <p>DERMOSCOPIC LESION ANALYSIS · RESNET-34 · GRAD-CAM ENABLED</p>
        """)

    # Main workspace
    with gr.Row(equal_height=False):

        # Left: upload + button
        with gr.Column(scale=1, min_width=280, elem_id="upload-col"):
            input_img = gr.Image(
                label="Dermoscopic Image",
                type="numpy",
                height=280,
            )
            run_btn = gr.Button("▶  RUN ANALYSIS", elem_id="run-btn", variant="primary")

            gr.HTML("""
            <div style="margin-top:12px;background:#0d1117;border:1px solid #21262d;
                        border-radius:8px;padding:14px 16px;font-family:'Space Mono',monospace;font-size:10px;color:#3d444d;line-height:1.7;">
              <span style="color:#555e6e;">ℹ MODEL INFO</span><br>
              Architecture · ResNet-34<br>
              Classes &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· 7 lesion types<br>
              Input &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· 224 × 224 px<br>
              XAI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Gradient-CAM
            </div>
            """)

        # Right: results
        with gr.Column(scale=2):
            diag_output = gr.HTML(
                value='<div style="font-family:\'Space Mono\',monospace;color:#3d444d;'
                      'padding:32px;text-align:center;font-size:12px;letter-spacing:1px;">'
                      'AWAITING IMAGE INPUT…</div>'
            )

            with gr.Row():
                cam_output = gr.Image(
                    label="Grad-CAM Activation Map",
                    height=224,
                    elem_id="cam-panel",
                )
                label_output = gr.Label(
                    num_top_classes=3,
                    label="Top-3 Confidence Ranking",
                )

    # Probability chart
    with gr.Row():
        plot_output = gr.Plot(label="Full Probability Distribution")

    # Disclaimer
    gr.HTML("""
    <div id="disclaimer">
      FOR RESEARCH AND EDUCATIONAL USE ONLY · NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL DIAGNOSIS ·
      ALWAYS CONSULT A QUALIFIED DERMATOLOGIST
    </div>
    """)

    # Event binding
    run_btn.click(
        fn=analyze_lesion,
        inputs=input_img,
        outputs=[diag_output, label_output, plot_output, cam_output],
    )

if __name__ == "__main__":
    demo.launch(share=False)