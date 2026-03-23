import torch
import numpy as np
import cv2
from model import SiameseUNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
def load_model():
    model = SiameseUNet(3, 1)
    checkpoint = torch.load("siamese_unet_checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# Preprocess
def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0   # 🔥 IMPORTANT
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0)
    return img.to(device)

# Predict
def predict(model, img1, img2):
    img1 = preprocess(img1)
    img2 = preprocess(img2)

    with torch.no_grad():
        output = model(img1, img2)

    return output.squeeze().cpu().numpy()