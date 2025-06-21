import torch, torch.nn as nn
import torch.nn.functional as F
import sys 

import torchvision.datasets as ds 
from torch.utils.data import DataLoader
from tqdm import tqdm 
import matplotlib.pyplot as plt 

from torchvision.transforms.v2 import (
    Compose, ToImage, ToDtype, Resize, Normalize
)
from torchvision.transforms.v2 import Lambda

from core_utils import (
    ImageEmbeddings,
    ViTBlock, copy_model_weights
)

SEED = 42
torch.manual_seed(SEED)

class CONFIG:
    P = 16 
    H =224
    W = 224
    D_IN = 768
    D_OUT = D_IN
    HEADS = 12
    LAYERS = 12
    HIDDEN_DIM = D_IN * 4
    CLASSES = 10 

device = (torch.accelerator.current_accelerator().type 
          if torch.accelerator.is_available() else "cpu")

class VisionTransformer(nn.Module):
    def __init__(self,CONFIG):
        super(VisionTransformer,self).__init__()
        self.image_embeddings = ImageEmbeddings(
            CONFIG.H, CONFIG.W,
            CONFIG.P,CONFIG.D_IN
        )
        self.vit_blocks = nn.Sequential(
            *[ViTBlock(CONFIG) 
              for _ in range(CONFIG.LAYERS)])
        self.norm = nn.LayerNorm(CONFIG.D_IN)
        self.out_linear = nn.Linear(CONFIG.D_IN,CONFIG.CLASSES)
    def forward(self,x):
        x = self.image_embeddings(x)
        x = self.vit_blocks(x)
        x = self.norm(x[:, 0])
        x = self.out_linear(x)
        return x

# in_ = torch.randn(1,3,224,224)

MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]

transforms = Compose([
    Resize((224, 224)),
    ToImage(),
    ToDtype(torch.float32, scale=True),
    Normalize(MEAN, STD),
])

test_dataset = ds.CIFAR10("../datasets",download=True, train=False,transform=transforms)
test_loader = DataLoader(test_dataset, batch_size=16,shuffle=True)

vit = VisionTransformer(CONFIG)

"""
class BaseConfig:
    H = 224
    W = 224
    HEADS = 16
    CLASSES = 10

model_size = lambda x: sum(p.numel() for p in x.parameters())
model_size_MB = lambda x: model_size(x) / 10**6
model_size_GB = lambda x: model_size(x) / 10**9
class L16Config(BaseConfig):
    P = 16
    D_IN = 1024
    D_OUT = D_IN
    LAYERS = 24
    HIDDEN_DIM = D_IN * 4
l_16_vit = VisionTransformer(L16Config)
print(f"ViT L/16 {model_size_MB(l_16_vit):.2f} MB")
del l_16_vit

class H14Config(BaseConfig):
    P = 14
    D_IN = 1280
    D_OUT = D_IN
    LAYERS = 32
    HIDDEN_DIM = D_IN * 4
h_14_vit = VisionTransformer(H14Config)
print(f"ViT H/14 {model_size_MB(h_14_vit):.2f} MB")
del h_14_vit

class g14Config(BaseConfig):
    P = 14
    D_IN = 1408
    D_OUT = D_IN
    LAYERS = 40
    HIDDEN_DIM = 6144


g_14_vit = VisionTransformer(g14Config)
print(f"ViT g/14 {model_size_GB(g_14_vit):.2f} GB")
del g_14_vit

class G14Config(BaseConfig):
    P = 14
    D_IN = 1664
    D_OUT = D_IN
    LAYERS = 48
    HIDDEN_DIM = 8192

G_14_vit = VisionTransformer(G14Config)
print(f"ViT G/14 {model_size_GB(G_14_vit):.2f} GB")
del G_14_vit
"""

weights = torch.load("../models/vit_b_16_pretrained_cifar10.pth",weights_only=True, map_location='cpu')
copy_model_weights(vit, weights)

vit = vit.to(device); vit.eval()

classes = ['airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck']
inv_transforms = Compose([
    Normalize(
      mean = [-i/j for i,j in zip(MEAN,STD)],
      std = [1/i for i in STD]),
    Lambda(lambda x: x * 255),
    Lambda(lambda x: x.permute(1,2,0)),
    Lambda(lambda x: x.to(torch.uint8))
])

iter_loader = iter(test_loader)
images, labels = next(iter_loader)

images = images.to(device)
with torch.no_grad():
    preds = vit(images)

for i, (image, label) in enumerate(zip(images,labels)):
    plt.subplot(4,4,i+1)
    pred = preds[i,:].argmax(dim=-1).item()
    cls = classes[pred]
    image = image.cpu().squeeze()
    inv_image = inv_transforms(image)
    plt.imshow(inv_image,aspect='auto')
    plt.axis('off')
    plt.title(f'{cls}',color='g' if pred == label else 'r')
plt.subplots_adjust(hspace=0.5,wspace=0.5)
plt.show()
