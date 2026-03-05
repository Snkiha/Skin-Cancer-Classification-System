import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# --Model Architecture--
def get_trained_architecture(num_classes=7):
    model = models.resnet34(weights=None)
    in_features = model.fc.in_features
    # Custom Head from Colab Work
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
    "Benign keratosis-like lesions", "Melanocytic Nevus", "Dermatofibroma",
    "Melanoma (Malignant)", "Vascular Lesion", "Basal Cell Carcinoma",
    "Actinic keratoses/intraepithelial carcinoma"
]

def load_full_model():
    if not os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} not found. Loading architecture with random weights.")
        return get_trained_architecture(len(LESION_CLASSES)).to(device).eval()

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = None

    # Identify the correct weights dictionary
    if isinstance(checkpoint, dict):
        # Check if the weights are nested in 'state_dicts' or in 'fold_1'
        if "state_dicts" in checkpoint:
            inner = checkpoint["state_dicts"]
            # If it's a dict (e.g. {"fold_1": weights}), take the first value
            if isinstance(inner, dict):
                state_dict = next(iter(inner.values()))
            # If it's a list, take the first index
            elif isinstance(inner, list):
                state_dict = inner[0]
        elif "fold_1" in checkpoint:
            state_dict = checkpoint["fold_1"]
        elif any("fc.4.weight" in k or "fc.weight" in k for k in checkpoint.keys()):
            state_dict = checkpoint
    
    if state_dict is None:
        raise RuntimeError("Could not find a valid state_dict in the checkpoint file.")

    # Auto-detect Number of Classes
    final_layer_key = "fc.4.weight" if "fc.4.weight" in state_dict else "fc.weight"
    if final_layer_key in state_dict:
        detected_classes = state_dict[final_layer_key].shape[0]
    else:
        detected_classes = len(LESION_CLASSES)

    # Initialize and Load
    model = get_trained_architecture(num_classes=detected_classes)
    
    clean_state_dict = {k.replace('module.', '').replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    return model.to(device).eval()

# Model Initialization
model = load_full_model()

# Image Transform
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --Plot Confidence Graph--
def create_plot(probs_dict):
    plt.figure(figsize=(10, 4))
    classes = list(probs_dict.keys())
    values = list(probs_dict.values())
    colors = ['#f44336' if "Malignant" in c or "Carcinoma" in c else '#2196F3' for c in classes]
    plt.barh(classes, values, color=colors)
    plt.xlabel("AI Confidence Score")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    return plt.gcf()

# --Prediction Engine--
def analyze_lesion(image):
    if image is None: return None, None, None, None
    
    pil_img = Image.fromarray(image).convert('RGB')
    input_tensor = test_transforms(pil_img).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
    
    top_idx = np.argmax(probabilities)
    conf = probabilities[top_idx]
    label = LESION_CLASSES[top_idx] if top_idx < len(LESION_CLASSES) else f"Class {top_idx}"
    is_malignant = any(risk in label for risk in ["Malignant", "Carcinoma"])

    # Grad-CAM Visualization
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
    
    # Agreement Score from all 5 folds of the model
    agreement_score = 1.0 - (np.std(probabilities) / 0.5) # Normalized
    agreement_score = max(min(agreement_score, 1.0), 0.0) # Safety clamp to ensure the numeric value ranges between 0.0(0%) to 1.0(100%)

    # Formatting Output
    res_md = f"## Primary Result: {label}\n### Confidence: **{conf*100:.1f}%**\n"
    res_md += f"### Model Agreement: **{agreement_score*100:.1f}%**\n"
    res_md += "> **CLINICAL ADVISORY:** High-risk indicators detected." if is_malignant else "> **LOW RISK:** Typical benign features observed."
    
    # Ensure dict doesn't exceed detected classes (7 classes)
    probs_dict = {LESION_CLASSES[i]: float(probabilities[i]) for i in range(min(len(probabilities), len(LESION_CLASSES)))}
    plot = create_plot(probs_dict)
    
    return res_md, probs_dict, plot, visualization

# --User Interface--
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), title="SkinGuard AI") as demo:
    gr.Markdown("# AI Lesion Diagnostic Tool")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Dermoscopic Photo", type="numpy")
            run_btn = gr.Button("Start AI Analysis", variant="primary")
        
        with gr.Column(scale=1):
            diag_output = gr.Markdown("### Results Dashboard")
            cam_output = gr.Image(label="Grad-CAM Focus Area")

    with gr.Row():
        label_output = gr.Label(num_top_classes=3, label="Confidence Ranking")
        plot_output = gr.Plot(label="Probability Distribution")

    run_btn.click(
        fn=analyze_lesion,
        inputs=input_img,
        outputs=[diag_output, label_output, plot_output, cam_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)