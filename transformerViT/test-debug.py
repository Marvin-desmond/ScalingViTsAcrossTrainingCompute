import torch, torch.nn as nn
from einops.layers.torch import Rearrange

from datasets import load_dataset 
import subprocess 
import sys 
from torchviz import make_dot

sys.path.append("../")
from transformerTiny.utils import printGreen, printOrange

from PIL import Image 
import numpy as np 
import matplotlib, matplotlib.pyplot as plt

from torchvision.models.vision_transformer import (
    vit_b_16, ViT_B_16_Weights,
    vit_b_32, ViT_B_32_Weights
)

extended_monitor = True
if extended_monitor:
    try:
        matplotlib.use('QtAgg')
    except ImportError: 
        subprocess.run("pip install PyQt6")
    except Exception as e:
        print(e)
        sys.exit(0)

from  torchvision.transforms.v2.functional import pil_to_tensor, resize

numpy_2_pil = lambda x: Image.fromarray(x)
pil_2_numpy = lambda x: np.asarray(x)
resize_fn = lambda x, size=240: resize(x, size=[size,size]).to(torch.float32)

dataset = load_dataset("huggingface/cats-image")
image = dataset['test']['image'][0]

# image = pil_2_numpy(image)
# image = torch.from_numpy(image.copy()).permute(2,0,1)
image = pil_to_tensor(image)
image = resize_fn(image)
C,H,W=image.shape
"""
conv = nn.Conv2d(3, 3*120*120, 10, 10, bias=False)
conv.weight = nn.Parameter(torch.ones_like(conv.weight))
with torch.no_grad():
    conv.weight.copy_(torch.ones_like(conv.weight))
"""

def extractPatch(patches, index, P):
    print(f"{patches.shape = }")
    patch = patches[...,index].view(1,3,P,P)
    patch = patch.squeeze(0).permute(1,2,0)
    return patch.to(torch.uint8).numpy()

def showPatches(patches, N, P):
    rows=cols=int(N**.5)
    for i in range(N):
        plt.subplot(rows,cols,i+1)
        patch_image = extractPatch(patches, i, P)
        plt.imshow(patch_image,aspect = "auto")
        plt.tight_layout()
        plt.xticks([]); plt.yticks([])
    plt.subplots_adjust(
    hspace=0.1,wspace=0.1 # wspace => aspect auto
    )
    plt.show()

def showImage(image: np.ndarray):
    plt.imshow(image)
    plt.show()

image = image.unsqueeze(0)
P=120
N=int((H*W)/(P**2))
assert H%N==0 and W%N==0, "image size must be integer multiple of patch"

unfold = nn.Unfold(kernel_size=(P, P),stride=P)
patches = unfold(image)
print(f"{patches.shape = }")
# showPatches(patches,N,P)
# einops for image patch extraction
rearr = Rearrange(
    'b c (h p1) (w p2) -> b (c p1 p2) (h w)', 
    p1 = P, p2 = P)

for name, child in rearr.named_children():
    print(f"{name = } ======> {child}\n\n")

patches_again = rearr(image)
print(f"{patches_again.shape = }")
# showPatches(patches_again,N,P)


# conv for image patches 
P=16
N=int((H*W)/P**2)
patch_layer=nn.Conv2d(3, 768, kernel_size=P,
stride=P)
patches_too = patch_layer(image)
print(patches_too.flatten(2).shape)
# showPatches(patches_too.flatten(2).transpose(1,2),N,P)


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer,self).__init__()
        self.layer = nn.Identity()
    def forward(self,x):
        return self.layer(x)
    
class ImageEmbedding(nn.Module):
    def __init__(self):
        super(ImageEmbedding,self).__init__()
        pass 
    def forward(self,x):
        return x 

in_ = torch.randn(1,3,226,256)
vit = VisionTransformer()

model = vit_b_16(ViT_B_16_Weights.DEFAULT)


# x = torch.randn(1, 3, 224, 224)
# y = model(x)  

# Visualize the graph
# make_dot(y, params=dict(model.named_parameters())).render("vit_graph", format="png")

# print(model)

for name, param in model.named_parameters():
    if "class" in name:
        printGreen(f"{name = } : {param.shape = }")
    else:
        printOrange(f"{name = } : {param.shape = }")

# for name, child in model.encoder.layers.named_modules():
#     print(f"{name = } ====> ")
# print(model)
