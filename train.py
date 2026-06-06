import os
import pickle
import numpy as np
import torch
import clip
from PIL import Image
from sklearn.linear_model import LogisticRegression

DATA_DIR = "data/train"
MODEL_PATH = "models/clip_classifier.pkl"

device = "cpu"

class_names = sorted(
    [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))],
    key=lambda x: int(x)
)
print(f"Classes: {len(class_names)}")

def extract_features(model_name):
    print(f"Loading {model_name}...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    feats = []
    for class_name in class_names:
        class_dir = os.path.join(DATA_DIR, class_name)
        for img_name in sorted(os.listdir(class_dir)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(class_dir, img_name)
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feat = feat.cpu().numpy().flatten()
            feats.append(feat)
    return np.array(feats, dtype=np.float32)

labels = np.array([int(c) for c in class_names for _ in range(
    len([f for f in os.listdir(os.path.join(DATA_DIR, c))
         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
)])

feats_b16 = extract_features("ViT-B/16")
feats_l14 = extract_features("ViT-L/14")

features = np.concatenate([feats_b16, feats_l14], axis=1)
print(f"Combined features: {features.shape}, labels: {labels.shape}")

print("Training classifier...")
clf = LogisticRegression(max_iter=3000, C=10.0, random_state=42, solver="lbfgs")
clf.fit(features, labels)

print(f"Training accuracy: {clf.score(features, labels):.4f}")

os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)
print(f"Saved classifier to {MODEL_PATH}")
