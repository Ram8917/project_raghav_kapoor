# predict.py
import torch
from model import SimpleUNet
from dataset import inferloader

def predict_temperature_classes(list_of_img_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleUNet(in_channels=1, out_classes=6)
    model.load_state_dict(torch.load("_checkpoints/final_weights.pth", map_location=device))
    model.to(device)
    model.eval()
    
    batch = inferloader(list_of_img_paths)
    batch = batch.to(device)
    
    with torch.no_grad():
        outputs = model(batch)
    
    predicted_classes = torch.argmax(outputs, dim=1)
    return predicted_classes.cpu().numpy()