import torch, torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import torch.optim as optim

import torchvision.datasets as ds 
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms.v2 import (
    Compose, ToImage, ToDtype, Resize, Normalize
)

from pathlib import Path 

SEED = 42
torch.random.manual_seed(SEED)

MODEL_DIR = Path("../models")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transforms = Compose([
    Resize((224, 224)),
    ToImage(), 
    ToDtype(torch.float32, scale=True),
    Normalize(MEAN, STD),
])

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# train_datasets = ds.CIFAR10("../datasets",download=True, train=True)
# train_datasets = ds.CIFAR10("../datasets",download=True, train=True,transform=transforms)
test_dataset = ds.CIFAR10("../datasets",download=True, train=False,transform=transforms)
test_loader = DataLoader(test_dataset, batch_size=16,shuffle=True)

model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False 
in_features, out_features = model.heads[0].in_features, 10
model.heads = nn.Sequential(nn.Linear(in_features, out_features))
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=.9)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7)


EPOCHS = 10
MODEL_PATH = MODEL_DIR / "saved_model.pth"
BEST_ACC = 0 
EPOCH_ACC = 0
model.train()


for _ in range(EPOCHS):
    LEN_DATA = len(test_dataset)
    LEN_LOADER = len(test_loader)
    LOSS = 0
    for images, labels in test_loader:
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        out_ints = torch.argmax(out.detach().cpu(),dim=-1)
        EPOCH_ACC += sum(labels.cpu() == out_ints) 
        LOSS += loss.item()
        loss = criterion(out, labels)
        optimizer.step() 
        break
    scheduler.step()
    EPOCH_ACC = EPOCH_ACC / LEN_DATA
    LOSS = LOSS / LEN_LOADER
    print(f" Acc: {EPOCH_ACC = }")
    print(f" Loss: {LOSS = }")

    break