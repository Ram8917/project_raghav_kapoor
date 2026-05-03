# predict.py
import torch
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from model import SimpleUNet
from dataset import inferloader

# ==========================================
# 1. REQUIRED BY THE AUTOMATED GRADER
# ==========================================
def predict_temperature_classes(list_of_img_paths):
    """
    Takes a list of image paths (provided by the grader), loads the best weights, 
    and returns a NumPy array of predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleUNet(in_channels=1, out_classes=6)
    
    # Safely load weights whether on GPU or CPU
    weights_path = "_checkpoints/best_weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    batch = inferloader(list_of_img_paths)
    batch = batch.to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        
    predicted_classes = torch.argmax(outputs, dim=1)
    return predicted_classes.cpu().numpy()

# ==========================================
# 2. LOCAL REPRODUCIBILITY TEST (For you and the TA)
# ==========================================
def calculate_single_miou(pred, target, num_classes=6):
    """Calculates mIoU for a single pair of 2D numpy arrays."""
    ious = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (target == cls)).sum()
        union = ((pred == cls) | (target == cls)).sum()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0

def test_unseen_validation_data():
    """Automatically finds the 20% unseen data, picks a day, and tests it."""
    print("🔍 Searching for local data to run a reproducibility test...")
    
    # EXACT FOLDER PATHS FOR YOUR REPOSITORY
    LWA_DIR = "./LWA_Corrected"
    LBL_DIR = "./Training_Labels_Global"
    
    lwa_files = sorted(glob.glob(os.path.join(LWA_DIR, "*.*")))
    lbl_files = sorted(glob.glob(os.path.join(LBL_DIR, "*.*")))
    
    if not lwa_files or not lbl_files:
        print(f"❌ Error: Could not find data in {LWA_DIR} or {LBL_DIR}. Skipping local test.")
        return

    # Recreate the exact 80/20 split so we only test UNSEEN data
    split_idx = int(0.8 * len(lwa_files))
    val_lwa_files = lwa_files[split_idx:]
    val_lbl_files = lbl_files[split_idx:]
    
    print(f"✅ Found {len(val_lwa_files)} unseen validation images.")
    
    # Pick one random unseen day
    random_idx = random.randint(0, len(val_lwa_files) - 1)
    test_lwa = val_lwa_files[random_idx]
    test_lbl = val_lbl_files[random_idx]
    print(f"🧪 Testing on specific day: {os.path.basename(test_lwa)}")
    
    # Use the grader's function to get the prediction!
    predictions = predict_temperature_classes([test_lwa])
    predicted_map = predictions[0] # Extract the single result
    
    # Load ground truth for comparison
    ground_truth = np.array(Image.open(test_lbl))
    
    # Calculate score
    miou_score = calculate_single_miou(predicted_map, ground_truth)
    print(f"⭐ Resulting Mean Intersection over Union (mIoU): {miou_score:.4f}")
    
    # Optional: Plot the result if matplotlib is available
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im1 = axes[0].imshow(ground_truth, cmap='turbo', vmin=0, vmax=5)
        axes[0].set_title("Actual Temperature Labels")
        axes[0].axis('off')

        im2 = axes[1].imshow(predicted_map, cmap='turbo', vmin=0, vmax=5)
        axes[1].set_title(f"Model Prediction (mIoU: {miou_score:.4f})")
        axes[1].axis('off')

        fig.colorbar(im1, ax=axes.ravel().tolist(), ticks=[0, 1, 2, 3, 4, 5], label="Temp Class")
        plt.suptitle("Unseen Data Validation Test", fontsize=16)
        plt.show()
    except Exception as e:
        print("Could not generate visual plot. Test complete.")

if __name__ == "__main__":
    test_unseen_validation_data()