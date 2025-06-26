import torch, torchvision
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import (
    Compose, ToImage, Resize, ToDtype, Normalize
)

from tqdm import tqdm
import sys
from pathlib import Path
from core_utils import VisionTransformer

SEED = 42; torch.manual_seed(SEED)

class BaseConfig:
    H = 224
    W = 224
    HEADS = 16
    CLASSES = 10
class L16Config(BaseConfig):
    P = 16
    D_IN = 1024
    D_OUT = D_IN
    LAYERS = 24
    HIDDEN_DIM = D_IN * 4

vit_l_16 = VisionTransformer(L16Config)

MEAN = [0.485,0.456,0.406]
STD = [0.229,0.224,0.225]
transforms = Compose([
    Resize((224, 224)), ToImage(),
    ToDtype(torch.float32, scale=True),
    Normalize(MEAN, STD),
])

train_data = torchvision.datasets.CIFAR10(
    root="./datasets", train=True, transform=transforms, download=True)
test_data = torchvision.datasets.CIFAR10(
    root="./datasets", train=False, transform=transforms, download=True)

from torch.utils.data import DataLoader
BATCH=32
train_loader = DataLoader(train_data,batch_size=BATCH,shuffle=True)
test_loader = DataLoader(test_data,batch_size=BATCH,shuffle=True)

optimizer = torch.optim.SGD(vit_l_16.parameters(),lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

device = (torch.accelerator.current_accelerator().type
          if torch.accelerator.is_available() else "cpu")
vit_l_16 = vit_l_16.to(device)

def test_metrics(model, loader, device):
  model.eval()
  running_loss = 0.0
  running_acc = 0.0
  total = 0
  for images, labels in tqdm(loader):
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
      outputs = model(images)
    loss = criterion(outputs,labels)
    preds = outputs.argmax(dim=-1)
    running_loss += loss.item()
    running_acc += (preds == labels).sum().item()
    total += labels.size(0)
  print(f"test loss: {running_loss / len(loader):.2f}")
  print(f"test acc: {(100 * running_acc / total):.2f} %")

# ONLY A SINGLE EPOCH

test_metrics(vit_l_16,test_loader,device)

vit_l_16.train()

import time
start = time.time()
running_loss = 0.0
for i, (images, labels) in enumerate(tqdm(train_loader)):
    images = images.to(device)
    actual = labels.to(device)
    optimizer.zero_grad()
    predicted = vit_l_16(images)
    loss = criterion(predicted,actual)
    running_loss += loss.item()
    if i % 200 == 0:
      last_loss = running_loss / 200
      print(' BATCH {} LOSS: {:.2f}'.format(i + 1, last_loss))
      running_loss = 0.
    loss.backward()
    optimizer.step()
stop = time.time()
print(f"time : {(stop - start) / 3600:.5f} hours per epoch")
