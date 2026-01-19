import torch
from flask import Flask, request, jsonify
from datetime import datetime
import logging
import os
from PIL import Image
import io
import base64
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask('unet_prediction')

# ======================
# UNet ARCHITECTURE (matches training)
# ======================

class DoubleConv(nn.Module):
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

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return torch.sigmoid(self.final_conv(x))  # sigmoid output

# ======================
# LOAD MODELS
# ======================

MODEL_DIR = "models"
horizontal_model = UNet()
ray_model = UNet()

def load_models():
    global horizontal_model, ray_model
    try:
        horizontal_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "horizontal_unet.pth"), map_location='cpu'))
        ray_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "unet_ray.pth"), map_location='cpu'))
        horizontal_model.eval()
        ray_model.eval()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

load_models()

# ======================
# HELPERS
# ======================

def tensor_to_base64(tensor):
    tensor = tensor.squeeze(0).clamp(0,1)  # remove batch dim
    if tensor.shape[0] == 1:  # single channel
        tensor = tensor.repeat(3,1,1)  # convert to RGB
    img = T.ToPILImage()(tensor)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ======================
# HEALTH CHECK
# ======================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'unet-prediction',
        'timestamp': str(datetime.now())
    })

# ======================
# PREDICTION ENDPOINT
# ======================

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        img = Image.open(file).convert('RGB')

        transform = T.Compose([
            T.Resize((256, 512)),
            T.ToTensor(),
        ])
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            h_pred = horizontal_model(input_tensor)
            r_pred = ray_model(input_tensor)

        return jsonify({
            'horizontal_unet': tensor_to_base64(h_pred),
            'unet_ray': tensor_to_base64(r_pred)
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

# ======================
# RUN APP
# ======================

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=9696)