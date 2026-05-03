# project_raghav_kapoor

# Upper Level Temperature Classification from Upper Local Wave Activity

**Author:** Raghav Kapoor  
**Institution:** IISER Pune  
**Task:** Semantic Segmentation (1-Channel Input $\rightarrow$ 6-Class Output)

---

## 🌍 Project Overview
This project investigates the structural relationship between mid-latitude atmospheric dynamics and surface temperature distributions. Using a custom Convolutional Neural Network (CNN), it performs semantic segmentation to map **Local Wave Activity (LWA)**—a diagnostic metric of jet stream meandering—to discrete **Surface Temperature Regimes** at higher levels of the atmosphere.

The model is trained to recognize the fuzzy spatial structures of atmospheric waves and classify each pixel into one of 6 temperature categories, demonstrating physical teleconnections between upper-level wave dynamics and regional weather.

---

## 📂 Directory Structure
The repository strictly follows the required grading nomenclature:
```text
project_raghav_kapoor/
│
├── _checkpoints/          # Directory for saved model weights
│   ├── best_weights.pth   # Checkpoint with highest validation mIoU
│
├── data/                  # Directory containing the dataset
│   ├── Input_LWA/         # 1-Channel 8-bit PNG LWA inputs (352x160)
│   └── Training_Labels/   # 1-Channel PNG integer masks (Classes 0-5)
│
├── config.py              # Hyperparameters and spatial dimensions
├── dataset.py             # PyTorch Dataset and DataLoader logic
├── model.py               # Custom SimpleUNet architecture
├── train.py               # Training loop, validation, and metrics
├── predict.py             # Inference and visualization script
├── interface.py           # Standardized bridge for automated grading
└── README.md              # Project documentation
```

## Data Preparation & Comparisons

### 1. The Input (Local Wave Activity)

- **Source:** ERA5 Reanalysis data (1996–2010 and 2020-2025), 250 hPa pressure level.
    
- **Computation:** LWA was pre-computed using the `falwa` (QGFieldNH18) framework.
    
- **Normalization:** To preserve the physical magnitude of atmospheric waves across different days, the continuous LWA floating-point data was normalized using a 2-Pass Global Minimum/Maximum approach across the entire 14-year dataset before being saved as 8-bit grayscale PNGs. The code I used for calculating the LWA images is mentioned in LWA_prep.py. To run that code you will need to install all the dependencies of the functions for calculating the Local Wave Activity. For reference visit , https://github.com/csyhuang/hn2016_falwa/  
- Look at the folder Code_Used_For_Data_Prep

### 2. The Output / Comparisons (Temperature Regimes)

- The target variables are discrete temperature classes at the 250 hPa level.
    
- Temperatures were discretized into 6 classes (0 through 5) based on specific physical thresholds (e.g., <210K, 210-217K, etc.). The code I used for preparing temp plots with categories is written in the Temp_prep.py file. You have the processed files in the data folder now which only required to be converted to tensors to be used for the training. So those codes are just for your reference to look at how I have pre-processed the data. 
    
- The model compares its predicted classification map against these ground-truth integer masks using Cross-Entropy Loss and evaluates physical accuracy using Mean Intersection over Union (mIoU).
    

---

##  Model Architecture

The project utilizes a custom **SimpleUNet**, designed specifically for lightweight semantic segmentation:

- **Encoder:** Progressively extracts features using DoubleConv blocks. I found the best channel configuration to use by trying the training for different cases and got this - 16, 32, 64 channel - with the maximum mIoU.
    
- **Decoder:** Uses `ConvTranspose2d` for upsampling.
    
- **Skip Connections:** Concatenates high-resolution encoder features with upsampled decoder features to preserve the precise spatial boundaries of weather systems.
    
- **Parameters:** ~482,534 trainable parameters, optimized for rapid training and inference.
    

---

## 🚀 How to Run the Code

### For Automated Grading (`interface.py`)

The project fully supports the automated grading pipeline. The `interface.py` file exposes all necessary variables and functions under the required standardized names:

- `TheModel` $\rightarrow$ `SimpleUNet`
    
- `the_trainer` $\rightarrow$ `train_model`
    
- `the_predictor` $\rightarrow$ `predict_temperature_classes`
    
- `TheDataset` $\rightarrow$ `ClimateDataset`
    
- `the_dataloader` $\rightarrow$ `get_dataloader`
    

_Note on the Training Signature:_ The `the_trainer` function maintains the strict required signature `train_model(model, num_epochs, train_loader, loss_fn, optimizer)`. Validation is handled safely via optional keyword arguments to prevent grader crashes.

### For Local Execution (VS Code)

To run the training loop locally with full metric logging (Training Loss, Validation Loss, mIoU):

1. Ensure your data is placed in `./data/Input_LWA/` and `./data/Training_Labels/`.
    
2. Install dependencies: `pip install torch torchvision numpy pillow`
    
3. Run the training script directly:
    

Bash

```
    python train.py

This will trigger the `__main__` execution block, automatically split the data (80/20 train/val), run the epochs, and save the best weights to the `_checkpoints` directory.
```

4. Clone the repository, navigate to the folder, and run the training script:
    
    Bash
    
    ```
    !git clone [https://github.com/Ram8917/project_raghav_kapoor.git](https://github.com/Ram8917/project_raghav_kapoor.git)
    %cd project_raghav_kapoor
    !python train.py
    ```

4. To check if the code is working fine, you can run predict.py to see how the model and real data look side by side 

```
python predict.py
```

One such example is as follows: 

![[Pasted image 20260503162147.png]]
