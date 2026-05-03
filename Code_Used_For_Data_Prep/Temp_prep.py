import xarray as xr
import numpy as np
from PIL import Image
import os
import time
from tqdm import tqdm
from datetime import timedelta

# --- 1. CONFIGURATION ---
input_dir = r'D:\Raghav\ERA5_Data'
output_dir = r'D:\Raghav\ERA5_Data\Training_Labels_Global'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

TARGET_SIZE = (352, 160) 

def discretize_temp(temp_array):
    labels = np.zeros_like(temp_array, dtype=np.uint8)
    labels[temp_array < 210] = 0
    labels[(temp_array >= 210) & (temp_array < 217)] = 1
    labels[(temp_array >= 217) & (temp_array < 225)] = 2
    labels[(temp_array >= 225) & (temp_array < 231)] = 3
    labels[(temp_array >= 231) & (temp_array < 236)] = 4
    labels[temp_array >= 236] = 5
    return labels

# --- 2. FILE GATHERING ---
years = list(range(1996, 2011)) + list(range(2020, 2026))
months = range(1, 13)
file_list = []

for y in years:
    for m in months:
        p = os.path.join(input_dir, f'era5_global_coarse_{y}_{m:02d}.nc')
        if os.path.exists(p):
            file_list.append(p)

print(f"🚀 Found {len(file_list)} months to process. Let's go!")
start_overall = time.time()

# --- 3. BATCH PROCESSING WITH PROGRESS BAR ---
for file_path in tqdm(file_list, desc="Overall Progress", unit="month"):
    try:
        with xr.open_dataset(file_path) as ds:
            # A. Standardize Coordinates & Memory
            ds = ds.sortby("latitude", ascending=True)
            ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby("longitude")
            ds = ds.compute() # Crucial for 0-360 alignment
            
            # B. Selection (Pressure level and Lat/Lon slice)
            ds_grid = ds.sel(
                latitude=slice(-79, 80), 
                longitude=slice(0, 351),
                pressure_level=250
            )
            
            t_field = ds_grid['t']
            date_str = os.path.basename(file_path).replace('era5_global_coarse_', '').replace('.nc', '')
            
            # C. Export Daily Images
            for i in range(len(t_field.valid_time)):
                day_data = t_field.isel(valid_time=i).values
                labeled_arr = discretize_temp(day_data)
                
                img = Image.fromarray(labeled_arr)
                img_resized = img.resize(TARGET_SIZE, resample=Image.NEAREST)
                
                # Using i+1 so day01 = 1st of the month
                img_name = f"label_{date_str}_day{i+1:02d}.png"
                img_resized.save(os.path.join(output_dir, img_name))
                
    except Exception as e:
        print(f"\n❌ Error in {os.path.basename(file_path)}: {e}")

# --- 4. FINAL STATS ---
end_overall = time.time()
elapsed = str(timedelta(seconds=int(end_overall - start_overall)))
print(f"\n🏁 Finished! Total Dataset Size: {len(os.listdir(output_dir))} images.")
print(f"⏱️ Total Time Taken: {elapsed}")