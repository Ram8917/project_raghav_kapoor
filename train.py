# train.py
import torch
import numpy as np
import os
import glob
from torch.optim.lr_scheduler import CosineAnnealingLR

def calculate_miou(outputs, targets, num_classes=6):
    preds = torch.argmax(outputs, dim=1)
    ious = []
    for cls in range(num_classes):
        inter = ((preds == cls) & (targets == cls)).sum().item()
        union = ((preds == cls) | (targets == cls)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """Handles one epoch of training."""
    model.train()
    epoch_loss = 0.0
    train_miou_sum = 0.0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        train_miou_sum += calculate_miou(outputs, targets)

    avg_loss = epoch_loss / len(dataloader)
    avg_miou = train_miou_sum / len(dataloader)
    return avg_loss, avg_miou

def validate_epoch(model, dataloader, loss_fn, device):
    """Handles one epoch of validation."""
    model.eval()
    val_loss_sum = 0.0
    val_miou_sum = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = loss_fn(outputs, targets)
            val_loss_sum += loss.item()
            val_miou_sum += calculate_miou(outputs, targets)
            
    avg_loss = val_loss_sum / len(dataloader)
    avg_miou = val_miou_sum / len(dataloader)
    return avg_loss, avg_miou

def train_model(model, num_epochs, train_loader, loss_fn, optimizer, val_loader=None, scheduler=None):
    """The master training loop with auto-saving checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 1. Create the folder BEFORE the loop starts
    os.makedirs("_checkpoints", exist_ok=True)
    best_val_miou = 0.0  # Track the highest score
    
    for epoch in range(num_epochs):
        train_loss, train_miou = train_epoch(model, train_loader, loss_fn, optimizer, device)
        
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        if val_loader:
            val_loss, val_miou = validate_epoch(model, val_loader, loss_fn, device)
            print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
            
            # 2. THE MAGIC AUTO-SAVE
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                torch.save(model.state_dict(), "_checkpoints/best_weights.pth")
                print(f"   ⭐ New best model saved! (mIoU: {best_val_miou:.4f})")
                
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}")
            # If no validation set, just save the latest epoch
            torch.save(model.state_dict(), "_checkpoints/latest_weights.pth")
            
    print("Training loop finished.")
    
    # Save final checkpoints
    os.makedirs("_checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "_checkpoints/final_weights.pth")
    print("Training complete. Weights saved.")

# ==========================================
# LOCAL EXECUTION & SETUP LOGIC
# ==========================================
if __name__ == "__main__":
    from model import SimpleUNet
    from dataset import get_dataloader
    from config import number_of_epochs, learning_rate
    
    # 1. Path Setup (Explicit 4-Folder Structure)
    TRAIN_LWA_DIR = "./data/Training_Dataset/LWA"
    TRAIN_LBL_DIR = "./data/Training_Dataset/Temperature"
    VAL_LWA_DIR = "./data/Validation_Dataset/LWA"
    VAL_LBL_DIR = "./data/Validation_Dataset/Temperature"
    
    # Grab all .png files
    train_lwa_files = sorted(glob.glob(os.path.join(TRAIN_LWA_DIR, "*.*")))
    train_lbl_files = sorted(glob.glob(os.path.join(TRAIN_LBL_DIR, "*.*")))
    
    val_lwa_files = sorted(glob.glob(os.path.join(VAL_LWA_DIR, "*.*")))
    val_lbl_files = sorted(glob.glob(os.path.join(VAL_LBL_DIR, "*.*")))
    
    # Ensure lists aren't empty
    if not train_lwa_files or not val_lwa_files:
        print("❌ Error: Could not find data files. Check your directory names.")
    else:
        # 2. Load directly into DataLoaders (No splitting required!)
        train_loader = get_dataloader(train_lwa_files, train_lbl_files, is_train=True)
        val_loader = get_dataloader(val_lwa_files, val_lbl_files, is_train=False)
        
        print(f"✅ Loaded {len(train_lwa_files)} training images and {len(val_lwa_files)} validation images.")
        
        # 3. Model & Device Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleUNet(in_channels=1, out_classes=6).to(device)
        
        # 4. Loss, Optimizer, and Scheduler
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=number_of_epochs)
        
        # 5. Execute Training
        print(f"🚀 Starting Local Training on {device}...")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Total trainable parameters in SimpleUNet: {total_params:,}")
        train_model(model, number_of_epochs, train_loader, loss_fn, optimizer, val_loader, scheduler)
