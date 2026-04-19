import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

# Class names
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Small CNN model
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load trained model
def load_trained_model(weights_path="fast_chest_model.pth", device="cpu", num_classes=2):
    device = torch.device(device)
    model = SmallCNN(num_classes=num_classes)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# Predict from PIL image
def predict_from_image(model, pil_image, device="cpu", class_names=None):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    img = pil_image.convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        label = class_names[idx] if class_names is not None else str(idx)
        confidence = float(probs[idx])

    return label, confidence, probs

# Check if image looks like chest X-ray + image stats
def image_stats_and_check(pil_image):
    img_gray = pil_image.convert("L")
    arr = np.array(img_gray).astype(np.float32)

    mean = float(arr.mean())
    std = float(arr.std())

    w, h = pil_image.size
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0

    edges = img_gray.filter(ImageFilter.FIND_EDGES)
    edge_count = int((np.array(edges) > 30).sum())

    is_xray = True
    if aspect_ratio < 0.5 or aspect_ratio > 1.6:
        is_xray = False
    if mean < 10 or mean > 240:
        is_xray = False

    stats = {
        "mean": mean,
        "std": std,
        "aspect_ratio": aspect_ratio,
        "edge_count": edge_count
    }

    return is_xray, stats