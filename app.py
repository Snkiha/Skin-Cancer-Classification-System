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
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ==============================================================================
# PREPROCESSING: DullRazor Hair Removal
# ==============================================================================

def dullrazor(image_bgr: np.ndarray) -> np.ndarray:
    """
    DullRazor algorithm for hair artefact removal in dermoscopic images.

    Steps:
      1. Convert to grayscale and apply a morphological black-hat transform to
         detect dark, thin structures (hairs) against a lighter background.
      2. Threshold the black-hat image to create a hair mask.
      3. Inpaint the masked pixels using the surrounding skin texture.

    Args:
        image_bgr: uint8 BGR image (H x W x 3).

    Returns:
        Hair-removed uint8 BGR image of the same shape.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Black-hat morphological transform: isolates dark structures smaller than
    # the structuring element (kernel_size should comfortably exceed hair width)
    kernel_size = 17
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to get binary hair mask; Otsu's method is adaptive to each image
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: dilate mask slightly to cover hair edges fully
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask = cv2.dilate(hair_mask, dil_kernel, iterations=1)

    # Inpaint: Telea's fast marching method fills masked areas from neighbours
    inpainted = cv2.inpaint(image_bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return inpainted, hair_mask


# ==============================================================================
# PREPROCESSING: HSV Colour-Space Lesion Segmentation
# ==============================================================================

def hsv_segment_lesion(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment the skin lesion from surrounding healthy skin using HSV colour
    analysis and morphological post-processing.

    Approach:
      - Convert to HSV; lesions typically differ from perilesional skin in
        saturation (S) and/or value (V).
      - Otsu-threshold the saturation channel to isolate the pigmented region.
      - Apply morphological closing to fill holes and opening to remove noise.
      - Keep only the largest connected component (the lesion body).
      - Apply a soft Gaussian-feathered mask so edges blend naturally.

    Returns:
        segmented_bgr : lesion on black background (uint8 BGR)
        mask_uint8    : binary mask (uint8, 0 / 255)
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # --- Saturation-channel Otsu threshold -----------------------------------
    # High saturation = pigmented lesion; low saturation = pale background skin
    _, sat_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Value-channel contribution (dark lesions on lighter skin) -----------
    # Invert value so darker areas become bright in the mask
    v_inv = 255 - v
    _, val_mask = cv2.threshold(v_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine: pixel is lesion if it is either highly saturated OR dark
    combined = cv2.bitwise_or(sat_mask, val_mask)

    # --- Morphological clean-up ----------------------------------------------
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_k, iterations=2)
    opened  = cv2.morphologyEx(closed,   cv2.MORPH_OPEN,  open_k,  iterations=1)

    # --- Keep largest connected component ------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels > 1:
        # stats[0] is background; pick the largest foreground component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask = np.where(labels == largest_label, np.uint8(255), np.uint8(0))
    else:
        clean_mask = opened  # fallback: no connected components found

    # --- Feathered edges via Gaussian blur -----------------------------------
    feathered = cv2.GaussianBlur(clean_mask, (21, 21), 0)

    # Apply mask to image
    mask_3ch   = feathered[:, :, np.newaxis].astype(np.float32) / 255.0
    segmented  = (image_bgr.astype(np.float32) * mask_3ch).astype(np.uint8)

    return segmented, clean_mask


# ==============================================================================
# FULL PREPROCESSING PIPELINE
# ==============================================================================

def preprocess_image(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full preprocessing pipeline on an RGB input image.

    Pipeline:
        RGB → BGR → DullRazor hair removal → HSV lesion segmentation → RGB output

    Returns:
        preprocessed_rgb  : fully preprocessed image ready for the CNN (RGB)
        hair_removed_rgb  : after DullRazor only (RGB), for visualisation
        segmented_rgb     : after segmentation (RGB), for visualisation
        hair_mask         : binary hair mask (uint8)
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Step 1 – DullRazor
    hair_removed_bgr, hair_mask = dullrazor(image_bgr)

    # Step 2 – HSV segmentation on the hair-removed image
    segmented_bgr, lesion_mask = hsv_segment_lesion(hair_removed_bgr)

    # Convert outputs back to RGB for PIL / display
    hair_removed_rgb = cv2.cvtColor(hair_removed_bgr, cv2.COLOR_BGR2RGB)
    segmented_rgb    = cv2.cvtColor(segmented_bgr,    cv2.COLOR_BGR2RGB)

    # The preprocessed image fed to the model: blend segmented region back onto
    # the hair-removed image at full opacity (keeps colour fidelity while
    # suppressing background variation)
    alpha       = lesion_mask.astype(np.float32)[:, :, np.newaxis] / 255.0
    blended_rgb = (segmented_rgb.astype(np.float32) * alpha +
                   hair_removed_rgb.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    return blended_rgb, hair_removed_rgb, segmented_rgb, hair_mask


# ==============================================================================
# MODEL ARCHITECTURE & LOADING  (unchanged from original)
# ==============================================================================

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


# ==============================================================================
# PLOT & ANALYSIS
# ==============================================================================

def create_plot(probs_dict):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    classes = list(probs_dict.keys())
    values  = list(probs_dict.values())
    colors  = ['#ff4d4d' if any(r in c for r in ["Malignant", "Carcinoma", "Actinic"])
               else '#00c8ff' for c in classes]

    bars = ax.barh(classes, values, color=colors, height=0.55, edgecolor='none')

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


def create_preprocessing_figure(original_rgb, hair_removed_rgb, segmented_rgb, hair_mask):
    """4-panel figure showing each preprocessing stage."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.patch.set_facecolor('#0d1117')

    panels = [
        (original_rgb,    "ORIGINAL"),
        (hair_removed_rgb, "HAIR REMOVED\n(DullRazor)"),
        (segmented_rgb,   "SEGMENTED\n(HSV)"),
        (hair_mask,       "HAIR MASK"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.set_facecolor('#0d1117')
        if img.ndim == 2:                       # grayscale mask
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title, color='#555e6e', fontfamily='monospace', fontsize=8,
                     letter_spacing=1, pad=6)
        ax.axis('off')

    plt.tight_layout(pad=1.0)
    return fig


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def analyze_lesion(image, use_hair_removal, use_hsv_segmentation):
    """
    Full analysis pipeline.

    Args:
        image               : numpy RGB array from Gradio image widget.
        use_hair_removal    : bool – apply DullRazor before inference.
        use_hsv_segmentation: bool – apply HSV segmentation before inference.
    """
    if image is None:
        return None, None, None, None, None

    original_rgb = image.copy()

    # ---- Selective preprocessing -------------------------------------------
    hair_removed_rgb = original_rgb
    hair_mask        = np.zeros(original_rgb.shape[:2], dtype=np.uint8)
    segmented_rgb    = original_rgb

    if use_hair_removal:
        image_bgr                      = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        hair_removed_bgr, hair_mask    = dullrazor(image_bgr)
        hair_removed_rgb               = cv2.cvtColor(hair_removed_bgr, cv2.COLOR_BGR2RGB)

    inference_input = hair_removed_rgb  # start from hair-removed (or original)

    if use_hsv_segmentation:
        bgr_for_seg                    = cv2.cvtColor(inference_input, cv2.COLOR_RGB2BGR)
        segmented_bgr, lesion_mask     = hsv_segment_lesion(bgr_for_seg)
        segmented_rgb                  = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)
        # Blend segmented back onto hair-removed for colour fidelity
        alpha          = lesion_mask.astype(np.float32)[:, :, np.newaxis] / 255.0
        inference_input = (segmented_rgb.astype(np.float32) * alpha +
                           inference_input.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    # ---- Build preprocessing visualisation panel ---------------------------
    prep_fig = create_preprocessing_figure(
        original_rgb, hair_removed_rgb, segmented_rgb, hair_mask
    )

    # ---- Model inference ---------------------------------------------------
    pil_img      = Image.fromarray(inference_input).convert('RGB')
    input_tensor = test_transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output        = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probabilities))
    conf    = probabilities[top_idx]
    label   = LESION_CLASSES[top_idx] if top_idx < len(LESION_CLASSES) else f"Class {top_idx}"
    is_malignant = any(risk in label for risk in ["Malignant", "Carcinoma", "Actinic"])

    # ---- Grad-CAM ----------------------------------------------------------
    target_layers = [model.layer4[-1]]
    targets       = [ClassifierOutputTarget(top_idx)]
    rgb_float     = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0

    try:
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        visualization = inference_input

    agreement_score = max(min(1.0 - (np.std(probabilities) / 0.5), 1.0), 0.0)

    # ---- Result HTML -------------------------------------------------------
    risk_icon   = "🔴" if is_malignant else "🟢"
    risk_label  = "HIGH RISK"  if is_malignant else "LOW RISK"
    risk_color  = "#ff4d4d"    if is_malignant else "#00e676"
    risk_banner = (
        f'<div style="background:#1a0a0a;border-left:4px solid {risk_color};padding:10px 16px;'
        f'border-radius:4px;margin-top:10px;font-family:monospace;">'
        f'<span style="color:{risk_color};font-weight:700;font-size:13px;">{risk_icon} {risk_label}</span>'
        f'<span style="color:#888;font-size:12px;margin-left:12px;">'
        + ("High-risk indicators detected — clinical review advised." if is_malignant
           else "Typical benign features observed.")
        + '</span></div>'
    )

    # Preprocessing badges
    def badge(active, label):
        bg  = "#0a2a1a" if active else "#161b22"
        col = "#00e676" if active else "#3d444d"
        bdr = "#00e676" if active else "#21262d"
        txt = "ON"      if active else "OFF"
        return (
            f'<span style="background:{bg};border:1px solid {bdr};color:{col};'
            f'padding:2px 8px;border-radius:3px;font-size:10px;margin-right:6px;">'
            f'{label} {txt}</span>'
        )

    prep_badges = badge(use_hair_removal, "DULLRAZOR") + badge(use_hsv_segmentation, "HSV SEG")

    res_md = f"""
<div style="font-family:'Courier New',monospace;background:#0d1117;padding:20px;border-radius:8px;border:1px solid #21262d;">
  <div style="margin-bottom:10px;">{prep_badges}</div>
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

    return res_md, probs_dict, plot, visualization, prep_fig


# ==============================================================================
# CSS
# ==============================================================================

css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

body, .gradio-container {
    background: #080c12 !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
}
.gr-box, .gr-form { box-shadow: none !important; }

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

label, .label-wrap span {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    color: #555e6e !important;
    text-transform: uppercase !important;
}

#upload-col .gr-image-upload {
    border: 1.5px dashed #21262d !important;
    border-radius: 10px !important;
    background: #0d1117 !important;
    transition: border-color 0.2s;
}
#upload-col .gr-image-upload:hover { border-color: #00c8ff !important; }

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
#run-btn:hover  { box-shadow: 0 0 32px rgba(14,165,233,0.45) !important; transform: translateY(-1px) !important; }
#run-btn:active { transform: translateY(0) !important; }

.gr-markdown, .gr-plot, .gr-label {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

#cam-panel img { border-radius: 8px !important; border: 1px solid #21262d !important; }
.gr-row    { gap: 16px !important; }
.gr-column { gap: 16px !important; }

/* Preprocessing toggle panel */
#prep-panel {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 16px;
    margin-top: 10px;
}

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

# ==============================================================================
# UI LAYOUT
# ==============================================================================

with gr.Blocks(title="SkinGuard AI") as demo:

    # Header
    with gr.Group(elem_id="app-header"):
        gr.HTML("""
        <h1>⬡ SKINGUARD AI</h1>
        <p>DERMOSCOPIC LESION ANALYSIS · RESNET-34 · GRAD-CAM · DULLRAZOR · HSV SEGMENTATION</p>
        """)

    # Main workspace
    with gr.Row(equal_height=False):

        # Left: upload + controls
        with gr.Column(scale=1, min_width=300, elem_id="upload-col"):
            input_img = gr.Image(
                label="Dermoscopic Image",
                type="numpy",
                height=280,
            )

            # ── Preprocessing toggles ────────────────────────────────────────
            gr.HTML('<div style="font-family:\'Space Mono\',monospace;font-size:10px;'
                    'color:#555e6e;letter-spacing:1.5px;margin:10px 0 4px;">PREPROCESSING</div>')

            with gr.Group(elem_id="prep-panel"):
                use_hair_removal = gr.Checkbox(
                    label="DullRazor Hair Removal",
                    value=True,
                    info="Morphological black-hat + Telea inpainting"
                )
                use_hsv_seg = gr.Checkbox(
                    label="HSV Lesion Segmentation",
                    value=True,
                    info="Saturation + value thresholding with morphological refinement"
                )

            run_btn = gr.Button("▶  RUN ANALYSIS", elem_id="run-btn", variant="primary")

            gr.HTML("""
            <div style="margin-top:12px;background:#0d1117;border:1px solid #21262d;
                        border-radius:8px;padding:14px 16px;font-family:'Space Mono',monospace;font-size:10px;color:#3d444d;line-height:1.9;">
              <span style="color:#555e6e;">ℹ MODEL INFO</span><br>
              Architecture · ResNet-34<br>
              Classes &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· 7 lesion types<br>
              Input &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· 224 × 224 px<br>
              XAI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· Gradient-CAM<br>
              <span style="color:#555e6e;">ℹ PREPROCESSING</span><br>
              Hair removal · DullRazor<br>
              Segmentation · HSV colour-space
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

    # ── Preprocessing pipeline visualisation ─────────────────────────────────
    with gr.Row():
        prep_plot_output = gr.Plot(label="Preprocessing Pipeline  ·  Original → Hair Removed → Segmented → Hair Mask")

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
        inputs=[input_img, use_hair_removal, use_hsv_seg],
        outputs=[diag_output, label_output, plot_output, cam_output, prep_plot_output],
    )


if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Default(primary_hue="blue"), css=css)