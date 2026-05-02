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
    """The master training loop required by the grader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        train_loss, train_miou = train_epoch(model, train_loader, loss_fn, optimizer, device)
        
        # Step the scheduler if one was provided
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Run validation if a validation loader was provided
        if val_loader:
            val_loss, val_miou = validate_epoch(model, val_loader, loss_fn, device)
            print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}")
    
    # Save checkpoints
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
    
    # 1. Path Setup (Update these to match your local folders)
    LWA_DIR = "./data/Input_LWA"
    LBL_DIR = "./data/Training_Labels_Global"
    
    # Grab all .png or .npy files
    lwa_files = sorted(glob.glob(os.path.join(LWA_DIR, "*.*")))
    lbl_files = sorted(glob.glob(os.path.join(LBL_DIR, "*.*")))
    
    # Ensure lists aren't empty before trying to split
    if not lwa_files or not lbl_files:
        print("❌ Error: Could not find data files. Check your LWA_DIR and LBL_DIR paths.")
    else:
        # 2. Data Splitting (80% Train, 20% Val)
        split_idx = int(0.8 * len(lwa_files))
        
        train_loader = get_dataloader(lwa_files[:split_idx], lbl_files[:split_idx], is_train=True)
        val_loader = get_dataloader(lwa_files[split_idx:], lbl_files[split_idx:], is_train=False)
        
        # 3. Model & Device Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleUNet(in_channels=1, out_classes=6).to(device)
        
        # 4. Loss, Optimizer, and Scheduler
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=number_of_epochs)
        
        # 5. Execute Training
        print(f"🚀 Starting Local Training on {device}...")
        train_model(model, number_of_epochs, train_loader, loss_fn, optimizer, val_loader, scheduler)