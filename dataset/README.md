# Dataset Folder - OHLCV Technical Analysis

This folder contains the datasets used for training and testing the deep learning models for chart pattern recognition (horizontal and ray support lines).

## Folder Structure

```
dataset/
├── train/
│   ├── full                # Full chart images or combined data for training
│   ├── input               # Input images used to train the models
│   ├── mask_horizontal     # Ground truth masks for horizontal support lines
│   ├── mask_ray            # Ground truth masks for ray support lines
│   ├── target_horizontal   # Target images corresponding to horizontal predictions
│   └── target_ray          # Target images corresponding to ray predictions
├── test/
│   └── input               # Input images for testing the trained models
└── README.md               # This file
```

## Description

- **train/full**: Contains the complete set of charts in their original format, used for generating training inputs and masks.  
- **train/input**: The input images fed into the models during training.  
- **train/mask_horizontal**: Binary masks showing horizontal support lines to supervise the horizontal model.  
- **train/mask_ray**: Binary masks showing ray support lines to supervise the ray model.  
- **train/target_horizontal**: Processed target images for the horizontal support prediction task.  
- **train/target_ray**: Processed target images for the ray support prediction task.  
- **test/input**: Images used to evaluate model performance after training.  

## Notes

- All images are derived from OHLCV chart data stored in the `data/` folder (JSON files).  
- Masks are binary images where highlighted pixels represent support levels.  
- Ensure that the `train/input` images correspond exactly to their masks and target images for correct training.  

## Usage

- During model training, `train/input` images are paired with either `mask_horizontal` or `mask_ray` depending on the model.  
- `test/input` images are used to generate predictions and visually compare them with expected support patterns.  