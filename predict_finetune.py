import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models, transforms
from PIL import Image

TEST_DIR = "data/test"
MODEL_PATH = "models/resnet50_finetuned.pth"
OUTPUT_PATH = "submission_finetune.csv"
NUM_CLASSES = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

test_files = sorted(
    [f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")],
    key=lambda x: int(x.replace(".jpg", ""))
)

predictions = []
with torch.no_grad():
    for i, fname in enumerate(test_files):
        image = Image.open(os.path.join(TEST_DIR, fname)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        pred = model(image).argmax(1).item()
        predictions.append(pred)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(test_files)}")

submission = pd.DataFrame({"ID": test_files, "Label": predictions})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {OUTPUT_PATH} with {len(predictions)} predictions")
