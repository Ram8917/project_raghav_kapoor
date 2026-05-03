import xarray as xr
import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm
from falwa.oopinterface import QGFieldNH18

# --- CONFIGURATION ---
input_dir = r"/mnt/d/Raghav/ERA5_Data"       # Where your .nc files are
temp_npy_dir = r"/mnt/d/Raghav/ERA5_Data/Temp_Raw_LWA" # Temp folder for raw calculation
output_png_dir = r"/mnt/d/Raghav/ERA5_Data/LWA_Corrected" # Final PNG destination

os.makedirs(temp_npy_dir, exist_ok=True)
os.makedirs(output_png_dir, exist_ok=True)

TARGET_PRESSURE = 250
HH = 7000.0  
KMAX = 17
DZ = 1000.0
HEIGHT = np.arange(0, KMAX) * DZ
TARGET_SIZE = (352, 160) # Width, Height (from your dataset config)

nc_files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))

if not nc_files:
    print(f"❌ No .nc files found in {input_dir}")
    exit()

# ==========================================
# PASS 1: Calculate LWA, Save Raw Arrays, Find Global Bounds
# ==========================================
global_max = -np.inf
global_min = np.inf

print("🚀 PASS 1: Computing QGFields and finding Global Min/Max...")
for file_path in nc_files:
    filename = os.path.basename(file_path).replace('.nc', '')
    print(f"\nProcessing {filename}...")
    
    with xr.open_dataset(file_path) as ds:
        # Standardize
        ds = ds.sortby("latitude", ascending=True)
        ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby("longitude")
        ds = ds.sortby("pressure_level", ascending=False)
        ds = ds.compute()

        xlon = ds.longitude.values
        ylat = ds.latitude.values
        plev = ds.pressure_level.values

        target_pseudoheight = -HH * np.log(TARGET_PRESSURE / plev[0])
        h_selected = np.argmin(np.abs(HEIGHT - target_pseudoheight))

        # Loop through each day in the current month's file
        for i in tqdm(range(len(ds.valid_time)), desc=f"Days in {filename}"):
            uu = ds['u'].isel(valid_time=i).values
            vv = ds['v'].isel(valid_time=i).values
            tt = ds['t'].isel(valid_time=i).values

            qgfield = QGFieldNH18(xlon, ylat, plev, uu, vv, tt, 
                                  northern_hemisphere_results_only=False, kmax=KMAX)
            qgfield.interpolate_fields()
            qgfield.compute_reference_states()
            qgfield.compute_lwa_and_barotropic_fluxes()

           # Extract the raw 2D array
            lwa_raw = qgfield.lwa[h_selected, :, :]
            
            # --- ADD YOUR SLICE HERE ---
            da_lwa = xr.DataArray(lwa_raw, coords=[ylat, xlon], dims=['lat', 'lon'])
            lwa_sliced = da_lwa.sel(lat=slice(-79, 80), lon=slice(0, 351)).values
            
            # 1. Update Global Bounds (using the sliced data)
            local_max = float(np.nanmax(lwa_sliced))
            local_min = float(np.nanmin(lwa_sliced))
            if local_max > global_max: global_max = local_max
            if local_min < global_min: global_min = local_min

            # 2. Save raw un-normalized sliced array to disk
            day_name = f"lwa_{filename}_day{i+1:02d}.npy"
            np.save(os.path.join(temp_npy_dir, day_name), lwa_sliced)

print(f"\n✅ PASS 1 Complete!")
print(f"🌍 Global Min: {global_min:.4f} | Global Max: {global_max:.4f}\n")

# ==========================================
# PASS 2: Normalize and Save as PNG
# ==========================================
print("🎨 PASS 2: Normalizing and creating PNGs...")

def normalize_to_uint8(lwa_array, g_min, g_max):
    clipped = np.clip(lwa_array, g_min, g_max)
    scaled = (clipped - g_min) / (g_max - g_min)
    return (scaled * 255).astype(np.uint8)

npy_files = sorted(glob.glob(os.path.join(temp_npy_dir, "*.npy")))

for npy_path in tqdm(npy_files, desc="Saving Final PNGs"):
    # Load the raw float array
    lwa_raw = np.load(npy_path)
    
    # Normalize it using the exact global bounds
    lwa_norm = normalize_to_uint8(lwa_raw, global_min, global_max)
    
    # Convert to grayscale Image and resize
    img = Image.fromarray(lwa_norm, mode='L')
    img_resized = img.resize(TARGET_SIZE, resample=Image.BILINEAR)
    
    # Save as PNG in the corrected folder
    png_name = os.path.basename(npy_path).replace('.npy', '.png')
    img_resized.save(os.path.join(output_png_dir, png_name))

print(f"\n🎉 All done! Normalized images are ready in: {output_png_dir}")