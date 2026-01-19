# Deep Learning Models for OHLCV Chart Pattern Recognition

This folder contains the trained deep learning models for predicting **support lines** in OHLCV chart images.  
Models are stored as `.pth` files compatible with PyTorch and can be loaded directly for inference.

![ChartPatternRecognition](../imgs/chart_pattern_recognition.jpeg)

## 📂 Models Included

1. **Horizontal Support Prediction**  
`horizontal_unet.pth`

- **Purpose:** Predicts horizontal support lines in OHLCV chart images.
- **Architecture:** U-Net convolutional neural network for image segmentation.
- **Input:** Chart image (PNG) resized to 512x256.
- **Output:** Mask highlighting horizontal support lines.
- **Use Case:** Detect stable price support levels for technical analysis.

2. **Ray Support Prediction**  
`unet_ray.pth`

- **Purpose:** Predicts diagonal/ray support lines in OHLCV chart images.
- **Architecture:** U-Net convolutional neural network for image segmentation.
- **Input:** Chart image (PNG) resized to 512x256.
- **Output:** Mask highlighting ray (diagonal) support lines.
- **Use Case:** Detect dynamic trend support lines for technical analysis.

## 📝 Notes

- Both models expect **normalized images** as input (`[0,1]` float format).
- Output masks are in the same dimensions as input images and can be visualized directly or used to extract coordinates.
- Ensure `torch` and `torchvision` are installed before loading models.
