import torch
import numpy as np
import medmnist
from medmnist import DermaMNIST
import torchvision.transforms as transforms
from app import load_full_model, LESION_CLASSES, device, test_transforms

def evaluate():
    print("Loading MedMNIST (DermaMNIST) dataset...")
    # download=True will download the dataset if not present
    dataset = DermaMNIST(split='test', download=True, size=28)
    
    print("Loading model...")
    model = load_full_model()
    
    medmnist_to_app = {
        0: 6, # akiec -> Actinic Keratosis
        1: 5, # bcc -> Basal Cell Carcinoma
        2: 0, # bkl -> Benign Keratosis
        3: 2, # df -> Dermatofibroma
        4: 3, # mel -> Melanoma
        5: 1, # nv -> Melanocytic Nevus
        6: 4  # vasc -> Vascular Lesion
    }
    
    results = []
    correct_count = 0
    total_images = 50
    
    # Selecting fixed 50 samples evenly distributed or random. Let's just pick the first 50 test samples.
    # To get a good mix, we can pick randomly using a fixed seed, or just first 50. First 50 might be heavily skewed to nv.
    # Let's collect exactly 50 images with a balanced class distribution if possible, or just random.
    np.random.seed(42)
    indices = np.random.choice(len(dataset), total_images, replace=False)
    
    print(f"Evaluating model on {total_images} external skin lesions...")
    
    for i, idx in enumerate(indices):
        img_pil, target = dataset[idx]
        actual_medmnist_label = int(target[0])
        actual_app_label_idx = medmnist_to_app[actual_medmnist_label]
        actual_class_name = LESION_CLASSES[actual_app_label_idx]
        
        # Apply transforms from app.py
        input_tensor = test_transforms(img_pil.convert('RGB')).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
            
        pred_idx = int(np.argmax(probabilities))
        pred_class_name = LESION_CLASSES[pred_idx] if pred_idx < len(LESION_CLASSES) else f"Class {pred_idx}"
        confidence = probabilities[pred_idx]
        
        is_correct = (pred_idx == actual_app_label_idx)
        if is_correct:
            correct_count += 1
            
        results.append({
            'sample_id': idx,
            'actual': actual_class_name,
            'predicted': pred_class_name,
            'confidence': confidence,
            'correct': is_correct
        })
        print(f"[{i+1}/{total_images}] Actual: {actual_class_name[:15]:15s} | Pred: {pred_class_name[:15]:15s} | Conf: {confidence:.2f} | {'Y' if is_correct else 'N'}")
        
    accuracy = correct_count / total_images
    print(f"\nEvaluation Complete! Accuracy: {accuracy*100:.1f}% ({correct_count}/{total_images})")
    
    # Generate Markdown Report
    report_content = f"# Model Validity Test Report\n\n"
    report_content += "## Overview\n"
    report_content += "A validity test was performed on the `ResNet-34` model using **50 independent external skin lesions** from the DermaMNIST dataset (HAM10000 source).\n\n"
    report_content += f"**Overall Accuracy:** {accuracy*100:.1f}% ({correct_count}/{total_images} correct)\n\n"
    
    report_content += "## Detailed Result per Lesion\n\n"
    report_content += "| Sample | Actual Diagnosis | Model Prediction | Confidence | Result |\n"
    report_content += "|:---:|:---|:---|:---:|:---:|\n"
    
    for i, res in enumerate(results):
        status = "✅ Correct" if res['correct'] else "❌ Incorrect"
        report_content += f"| {i+1} | {res['actual']} | {res['predicted']} | {res['confidence']*100:.1f}% | {status} |\n"
        
    report_content += "\n## Analysis\n"
    report_content += "The external dataset consists of 28x28 images, while the model expects 224x224 input. A drop in performance compared to original resolution validation sets is expected due to upscaling interpolation artifacts, however, this serves as a baseline robust check for the model's feature extraction generalization on external downscaled samples.\n"
    
    with open("C:/Users/kisha/.gemini/antigravity/brain/f1789d37-8989-435f-b2d6-5262ef613491/model_evaluation_report.md", "w", encoding='utf-8') as f:
        f.write(report_content)
        
    print("Report saved to artifacts directory.")

if __name__ == '__main__':
    evaluate()
