#!/usr/bin/env python
# coding: utf-8

# # 0. Importing the libs and read the data
# Core Libraries
from PIL import Image
import pandas as pd
import numpy as np
import traceback
import json
import glob
import cv2
import os

# Plotting Libraries
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from collections import Counter
from statistics import mode
import mplfinance as mpf

# Machine Learning Libraries
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import torch

# # 1. Data preparation and data cleaning
# ### Loading the JSON files
dataset_base_path = "./data/"
# Find all JSON files in the specified path using glob
json_files = glob.glob(os.path.join(dataset_base_path, "*.json"))
json_files.sort() # Sort for consistent ordering if needed

num_files = len(json_files)
print(f"Found {num_files} JSON files in: {dataset_base_path}")

if num_files == 0:
    print("\n--- WARNING ---")
    print("No JSON files found. Please ensure the 'dataset_base_path' is correct and points to the folder containing the JSON files.")
else:
    print("\nFirst 5 sample file names:")
    for f in json_files[:min(5, num_files)]:
        print(f" - {os.path.basename(f)}")


# ### Converted all JSON keys to lowercase
# - I did this because I see that there are some columns keys that contain uppercase letters and this confused me in my analysis
def lowercase_keys(obj):
    """Recursively convert all dictionary keys to lowercase"""
    if isinstance(obj, dict):
        return {key.lower(): lowercase_keys(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [lowercase_keys(item) for item in obj]
    else:
        return obj

files_changed = 0
for file_path in json_files:
    try:
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Convert all keys to lowercase
        lowercase_data = lowercase_keys(data)

        # Check if data was changed
        if lowercase_data != data:
            files_changed += 1
            # Write back to file
            with open(file_path, 'w') as f:
                json.dump(lowercase_data, f, indent=2)
            print(f"{os.path.basename(file_path)} (modified)")    
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
print(f"{files_changed}/{len(json_files)} file changes")

# ### - Converting the `start_date` and `end_date` columns that are outside the range of the chandles stick to max and min values from the chandles date range
# ### - Converting the `price` that is outside the range [`low`, `high`] to the lowest or the highes value
for file_path in json_files:
    with open(file_path, 'r') as f:
        data = json.load(f)

    ohlcv = data['ohlcv_data']
    labels = data['labels']

    dates = [pd.to_datetime(c['time'], utc=True) for c in ohlcv]
    prices = [float(c[k]) for c in ohlcv for k in ['open', 'high', 'low', 'close']]
    min_d, max_d = min(dates), max(dates)
    min_p, max_p = min(prices), max(prices)

    # Check horizontal lines
    for hl in labels.get('horizontal_lines', []):
        if not (min_p <= float(hl['price']) <= max_p):
            hl['price'] = max(min_p, min(float(hl['price']), max_p))

    # Check and fix ray lines
    for rl in labels.get('ray_lines', []):
        start_d = pd.to_datetime(rl['start_date'], utc=True)
        end_d = pd.to_datetime(rl['end_date'], utc=True)

        if not (min_d <= start_d <= max_d):        
            start_d = max(min_d, min(start_d, max_d))
        if not (min_d <= end_d <= max_d):
            end_d = max(min_d, min(end_d, max_d))

        rl['start_date'] = start_d.isoformat()
        rl['end_date'] = end_d.isoformat()

    # Save fixed data
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


# ### Check if there are missing lines in the image of the chart
# - This will be the data that will enter to test
empty_h_files = 0
empty_r_files = 0
empty_both_files = 0
files_empty_h = []
files_empty_r = []
files_empty_both = []

for file_path in json_files:
    with open(file_path, 'r') as f:
        data = json.load(f)

    labels = data.get("labels", {})

    h_lines = labels.get("horizontal_lines", [])
    r_lines = labels.get("ray_lines", [])

    filename = os.path.basename(file_path)

    if not h_lines:
        empty_h_files += 1
        files_empty_h.append(filename)

    if not r_lines:
        empty_r_files += 1
        files_empty_r.append(filename)

    if not h_lines and not r_lines:
        empty_both_files += 1
        files_empty_both.append(filename)

files_with_any_empty = set(files_empty_h) | set(files_empty_r)
print("Empty label statistics:")
print(f"Files with EMPTY horizontal_lines: {empty_h_files}/{len(json_files)} -- {100 * empty_h_files / len(json_files):.2f}%")
print(f"Files with EMPTY ray_lines:        {empty_r_files}/{len(json_files)} - {100 * empty_r_files / len(json_files):.2f}%")
print(f"Files with BOTH empty:             {empty_both_files}/{len(json_files)} -- {100 * empty_both_files / len(json_files):.2f}%")
print(f"Total with AT LEAST ONE EMPTY:     "f"{len(files_with_any_empty)}/{len(json_files)} - {100 * len(files_with_any_empty) / len(json_files):.2f}%")


# ### Loading the train data
# - Helper function to load data train and test data
def loading_json_to_png_chart_training(
    input_folder,
    target_horizontal_folder,
    target_ray_folder,
    full_folder,
    j_files
):
    # ---- Ensure output folders exist ----
    for folder in [input_folder, target_horizontal_folder, target_ray_folder, full_folder]:
        os.makedirs(folder, exist_ok=True)

    for idx, sample_file_path in enumerate(j_files):
        print(f"\n[{idx+1}/{len(j_files)}] Processing: {os.path.basename(sample_file_path)}")

        try:
            with open(sample_file_path, "r") as f:
                data = json.load(f)

            ohlcv_data = data.get("ohlcv_data", [])
            labels = data.get("labels", {})
            horizontal_lines = labels.get("horizontal_lines", [])
            ray_lines = labels.get("ray_lines", [])

            # ---- Convert OHLCV to DataFrame ----
            df = pd.DataFrame(ohlcv_data)
            if df.empty:
                print("Empty OHLCV data, skipping.")
                continue

            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.dropna(subset=["open", "high", "low", "close"], inplace=True)

            if df.empty:
                print("DataFrame empty after cleaning, skipping.")
                continue

            df["volume"] = df["volume"].fillna(0).astype(np.int64)

            # ---- Prepare horizontal lines ----
            hlines_plot = [
                float(hl["price"]) for hl in horizontal_lines if "price" in hl
            ]

            # ---- Prepare ray lines ----
            alines_plot = []
            for rl in ray_lines:
                try:
                    start_dt = pd.to_datetime(rl["start_date"])
                    end_dt = pd.to_datetime(rl["end_date"])
                    start_p = float(rl["start_price"])
                    end_p = float(rl["end_price"])
                    alines_plot.append([(start_dt, start_p), (end_dt, end_p)])
                except Exception as e:
                    print(f"Warning: skipping ray line {rl}: {e}")

            base_name = os.path.basename(sample_file_path).replace(".json", "")

            # ---- Common plot settings ----
            base_plot_kwargs = dict(
                type="candle",
                volume=True,
                style="yahoo",
                ylabel="Price",
                ylabel_lower="Volume",
                figratio=(14, 7),
                figscale=1.0
            )

            # ========== 1 Candles ONLY ==========
            mpf.plot(
                df,
                savefig=os.path.join(input_folder, f"{base_name}.png"),
                **base_plot_kwargs
            )

            # ========== 2 Candles + Horizontal ==========
            if hlines_plot:
                plot_kwargs = dict(base_plot_kwargs)
                plot_kwargs["hlines"] = dict(
                    hlines=hlines_plot,
                    colors="blue",
                    linestyle="--",
                    linewidths=1
                )

                mpf.plot(
                    df,
                    savefig=os.path.join(target_horizontal_folder, f"{base_name}.png"),
                    **plot_kwargs
                )

            # ========== 3 Candles + Ray ==========
            if alines_plot:
                plot_kwargs = dict(base_plot_kwargs)
                plot_kwargs["alines"] = dict(
                    alines=alines_plot,
                    colors="red",
                    linestyle=":",
                    linewidths=1.5
                )

                mpf.plot(
                    df,
                    savefig=os.path.join(target_ray_folder, f"{base_name}.png"),
                    **plot_kwargs
                )

            # ========== 4 Candles + Horizontal + Ray ==========
            if hlines_plot or alines_plot:
                plot_kwargs = dict(base_plot_kwargs)

                if hlines_plot:
                    plot_kwargs["hlines"] = dict(
                        hlines=hlines_plot,
                        colors="blue",
                        linestyle="--",
                        linewidths=1
                    )

                if alines_plot:
                    plot_kwargs["alines"] = dict(
                        alines=alines_plot,
                        colors="red",
                        linestyle=":",
                        linewidths=1.5
                    )

                mpf.plot(
                    df,
                    savefig=os.path.join(full_folder, f"{base_name}.png"),
                    **plot_kwargs
                )

            print("✔ Charts saved successfully.")

        except Exception as e:
            print(f"Error processing {sample_file_path}: {e}")
            print(traceback.format_exc())

def loading_json_to_png_chart_test(output_folder, json_files):
    for idx, sample_file_path in enumerate(json_files[:len(json_files)]):
        print(f"\n[{idx+1}/{len(json_files)}] Processing: {os.path.basename(sample_file_path)}")

        try:
            with open(sample_file_path, 'r') as f:
                data = json.load(f)

            ohlcv_data = data.get('ohlcv_data', [])
            labels = data.get('labels', {})
            horizontal_lines = labels.get('horizontal_lines', [])
            ray_lines = labels.get('ray_lines', [])

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

            if df.empty:
                print("DataFrame empty after cleaning, skipping this file.")
                continue

            # Prepare horizontal lines
            hlines_plot = [float(hl['price']) for hl in horizontal_lines if 'price' in hl]

            # Prepare ray lines
            alines_plot = []
            for rl in ray_lines:
                try:
                    start_dt = pd.to_datetime(rl.get('start_date'))
                    end_dt = pd.to_datetime(rl.get('end_date'))
                    start_p = float(rl.get('start_price'))
                    end_p = float(rl.get('end_price'))

                    # Align timezone if needed
                    if df.index.tz is not None:
                        if start_dt.tz is None: start_dt = start_dt.tz_localize(df.index.tz)
                        else: start_dt = start_dt.tz_convert(df.index.tz)
                        if end_dt.tz is None: end_dt = end_dt.tz_localize(df.index.tz)
                        else: end_dt = end_dt.tz_convert(df.index.tz)

                    alines_plot.append([(start_dt, start_p), (end_dt, end_p)])
                except Exception as e:
                    print(f"  - Warning: Error processing ray line {rl}: {e}")

            # Chart title and save path
            base_name = os.path.basename(sample_file_path).replace('.json', '')
            chart_title = f"Sample Chart {idx+1}: {base_name}"
            save_path = os.path.join(output_folder, f"{base_name}.png")

            plot_kwargs = dict(
                type='candle',
                volume=True,
                style='yahoo',
                title=chart_title,
                ylabel='Price',
                ylabel_lower='Volume',
                figratio=(14, 7),
                figscale=1.0,
                hlines=dict(hlines=hlines_plot, colors='blue', linestyle='--', linewidths=1) if hlines_plot else None,
                alines=dict(alines=alines_plot, colors='red', linestyle=':', linewidths=1.5) if alines_plot else None,
                savefig=save_path
            )
            # Remove None kwargs
            plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}

            # Ensure volume is integer
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(np.int64)

            mpf.plot(df, **plot_kwargs)
            print(f"✔ Charts saved successfully.")

        except Exception as e:
            print(f"Error processing file {sample_file_path}: {e}")
            print(traceback.format_exc())


# ### Load all the charts
# ### Load only the charts that have horizontal and ray lines 
# Loading only the training data (json_files - files_empty_h - files_empty_r - files_empty_both)
loading_json_to_png_chart_training(
    input_folder = "dataset/train/input",
    target_horizontal_folder = "dataset/train/target_horizontal",
    target_ray_folder = "dataset/train/target_ray",
    full_folder = "dataset/train/full",
    j_files = list(set(json_files) - set('./data/' + i for i in files_with_any_empty))
)


# ### Load the charts that do not have horizontal or ray lines
# Loading only the test data (files_with_any_empty)
loading_json_to_png_chart_test("dataset/test/input", list('./data/' + i for i in files_with_any_empty))


# # Data Preparation (imgs)
# ### Device used
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ### Define dataset paths
DATASET_ROOT = "dataset/train"
INPUT_DIR = os.path.join(DATASET_ROOT, "input")
HORIZONTAL_DIR = os.path.join(DATASET_ROOT, "target_horizontal")
RAY_DIR = os.path.join(DATASET_ROOT, "target_ray")

print("Input images:", len(glob(INPUT_DIR + "/*.png")))
print("Horizontal targets:", len(glob(HORIZONTAL_DIR + "/*.png")))
print("Ray targets:", len(glob(RAY_DIR + "/*.png")))


# ### Verify filename alignment
input_files = sorted(glob(INPUT_DIR + "/*.png"))
h_files = sorted(glob(HORIZONTAL_DIR + "/*.png"))
r_files = sorted(glob(RAY_DIR + "/*.png"))

input_names = [os.path.basename(f) for f in input_files]
h_names = [os.path.basename(f) for f in h_files]
r_names = [os.path.basename(f) for f in r_files]

assert input_names == h_names == r_names, "❌ File names do not match!"
print("✅ File alignment OK")


# ### Define image transforms
IMAGE_SIZE = (256, 512)  # (height, width)

image_transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),  # converts to [0,1]
])

mask_transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),  # converts to [0,1]
])

# ### Custom Dataset class
class ChartLineDataset(Dataset):
    """Dataset for chart images and their line masks"""
    def __init__(self, input_dir, target_dir, transform):
        self.input_files = sorted(glob.glob(input_dir + "/*.png"))
        self.target_files = sorted(glob.glob(target_dir + "/*.png"))
        self.transform = transform

        assert len(self.input_files) == len(self.target_files), \
            "Input and target counts do not match"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load images
        x = Image.open(self.input_files[idx]).convert("RGB")
        y = Image.open(self.target_files[idx]).convert("L")  # grayscale mask

        # Apply transforms
        x = self.transform(x)
        y = self.transform(y)

        # Force binary mask
        y = (y > 0.1).float()

        return x, y


# ### Instantiate datasets
# Create datasets
transform = T.Compose([
    T.Resize((256, 512)),
    T.ToTensor()
])

# Horizontal line dataset
horizontal_ds = ChartLineDataset(
    input_dir="dataset/train/input",
    target_dir="dataset/train/mask_horizontal",
    transform=transform
)

# Ray line dataset
ray_ds = ChartLineDataset(
    input_dir="dataset/train/input",
    target_dir="dataset/train/mask_ray",
    transform=transform
)

print("Horizontal samples:", len(horizontal_ds))
print("Ray samples:", len(ray_ds))


# ### DataLoaders
# Create DataLoaders
BATCH_SIZE = 4

# Horizontal line DataLoader
horizontal_loader = DataLoader(
    horizontal_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# Ray line DataLoader
ray_loader = DataLoader(
    ray_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ### Create mask output folders
MASK_ROOT = "dataset/train"

MASK_HORIZONTAL_DIR = os.path.join(MASK_ROOT, "mask_horizontal")
MASK_RAY_DIR = os.path.join(MASK_ROOT, "mask_ray")

os.makedirs(MASK_HORIZONTAL_DIR, exist_ok=True)
os.makedirs(MASK_RAY_DIR, exist_ok=True)

print("Mask folders ready")


# ### Define color thresholds
# OpenCV uses BGR, not RGB
BLUE_LOWER = np.array([200, 0, 0])
BLUE_UPPER = np.array([255, 80, 80])

RED_LOWER = np.array([0, 0, 200])
RED_UPPER = np.array([80, 80, 255])


# ### Generate masks
target_horizontal_files = sorted(glob.glob(HORIZONTAL_DIR + "/*.png"))
target_ray_files = sorted(glob.glob(RAY_DIR + "/*.png"))

for h_path, r_path in tqdm(zip(target_horizontal_files, target_ray_files),
                           total=len(target_horizontal_files)):

    filename = os.path.basename(h_path)

    # Load images
    img_h = cv2.imread(h_path)
    img_r = cv2.imread(r_path)

    # Create binary masks
    mask_h = cv2.inRange(img_h, BLUE_LOWER, BLUE_UPPER)
    mask_r = cv2.inRange(img_r, RED_LOWER, RED_UPPER)

    # Save masks
    cv2.imwrite(os.path.join(MASK_HORIZONTAL_DIR, filename), mask_h)
    cv2.imwrite(os.path.join(MASK_RAY_DIR, filename), mask_r)


# ### Sanity-check tensors
# **What this means:**
# - 3 ---------> RGB channels
# - 1 ---------> binary mask channel
# - 256 x 512 -> resized image resolution
x, y = horizontal_ds[0]

print(x.shape)      # [3, H, W]
print(y.shape)      # [1, H, W]
print(y.unique())   # tensor([0., 1.])


# ### Define the U-Net model
class DoubleConv(nn.Module):
    """
    Two consecutive convolution layers + BatchNorm + ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# Cell 3: Define the full U-Net
class UNet(nn.Module):
    """
    U-Net architecture for image 
    segmentation.
    """

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Final 1x1 convolution to map to output channel (mask)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse for upsampling

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # upsample
            skip_connection = skip_connections[idx//2]

            # If input size is odd, pad
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return torch.sigmoid(self.final_conv(x))  # Sigmoid for binary mask


# ### Check the model
model = UNet(in_channels=3, out_channels=1).to(device)
print(model)

x = torch.randn((4, 3, 256, 512)).to(device)  # batch of 4 images
with torch.no_grad():
    y = model(x)

print(y.shape)  # Should be [4, 1, 256, 512]
print(y.min(), y.max())


# # Prepare training and validation loops for horizontal line model
# **Folder used:**
# - dataset/train/input
# - dataset/train/mask_horizontal
# 
# ### Define loss function
# 
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return self.bce(preds, targets) + self.dice(preds, targets)


# ### Fine-tunning: Optimizer & learning rate
# **Why 1e-4?**
# - stable
# - works extremely well for U-Net
# - default in segmentation papers
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = BCEDiceLoss()


# ### Training loop
# **Now we train horizontal-line model only**
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ### Validation loop
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()

    return total_loss / len(loader)


# ### Training lool (Horizontal Model)
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        model,
        horizontal_loader,
        optimizer,
        criterion,
        device
    )

    print(f"Epoch {epoch+1}/{num_epochs} | loss = {train_loss:.4f}")


# ### Save the model
torch.save(model.state_dict(), "models/horizontal_unet.pth")


# ### Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("models/horizontal_unet.pth", map_location=device))
model.eval()


# ### Load one test image
transform = T.Compose([
    T.Resize((256, 512)),
    T.ToTensor()
])

img_path = sorted(glob.glob("dataset/test/input/*.png"))[17]

img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)


# ### Predict mask
with torch.no_grad():
    pred = model(x)

pred = pred.squeeze(0).squeeze(0) # [256, 512]


# ### Threshold prediction
mask = (pred > 0.5).float()

# # Prepare training and validation loops for ray line model
# **Folder used:**
# - dataset/train/input
# - dataset/train/mask_ray
ray_model = UNet(in_channels=3, out_channels=1).to(device)
optimizer_ray = torch.optim.Adam(ray_model.parameters(), lr=1e-4)
criterion_ray = BCEDiceLoss()


# ### Training loop (Ray Model)
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        ray_model,
        ray_loader,
        optimizer_ray,
        criterion_ray,
        device
    )
    print(f"Epoch {epoch+1}/{num_epochs} | Ray loss = {train_loss:.4f}")


# ### Save model
os.makedirs("models", exist_ok=True)
torch.save(ray_model.state_dict(), "models/unet_ray.pth")


# ### Load the trained model
ray_model = UNet(in_channels=3, out_channels=1).to(device)
ray_model.load_state_dict(torch.load("models/unet_ray.pth", map_location=device))
ray_model.eval()
