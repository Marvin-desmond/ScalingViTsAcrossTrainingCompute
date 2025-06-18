import torch, torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self):
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

vit = VisionTransformer()

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

images, labels = next(iter(test_loader))
for i, (image, label) in enumerate(zip(images,labels)):
    image = image.to(device)
    image = image.unsqueeze(0)
    plt.subplot(4,4,i+1)
    with torch.no_grad():
        pred = vit(image)
    pred = pred.argmax(dim=-1).item()
    cls = classes[pred]
    image = image.cpu().squeeze()
    inv_image = inv_transforms(image)
    plt.imshow(inv_image,aspect='auto')
    plt.axis('off')
    plt.title(f'{cls}',color='g' if pred == label else 'r')
plt.subplots_adjust(hspace=0.5,wspace=0.5)
plt.show()
