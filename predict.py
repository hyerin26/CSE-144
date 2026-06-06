import os
import pickle
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image

TEST_DIR = "data/test"
MODEL_PATH = "models/clip_classifier.pkl"
OUTPUT_PATH = "submission.csv"

device = "cpu"

test_files = sorted(
    [f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")],
    key=lambda x: int(x.replace(".jpg", ""))
)

with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

def get_features(model_name):
    print(f"Loading {model_name}...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    feats = []
    with torch.no_grad():
        for i, file_name in enumerate(test_files):
            img_path = os.path.join(TEST_DIR, file_name)
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat.cpu().numpy().flatten())
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(test_files)}")
    return np.array(feats, dtype=np.float32)

feats_b16 = get_features("ViT-B/16")
feats_l14 = get_features("ViT-L/14")
features = np.concatenate([feats_b16, feats_l14], axis=1)

predictions = clf.predict(features).tolist()

submission = pd.DataFrame({"ID": test_files, "Label": predictions})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {OUTPUT_PATH} with {len(predictions)} predictions")
