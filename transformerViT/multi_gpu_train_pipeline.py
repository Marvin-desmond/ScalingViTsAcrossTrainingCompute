import torch, torchvision 
import os, sys 
import torch.distributed as dist
import torch.multiprocessing as mp 
from torch.utils.data import DataLoader, Subset, random_split 
from random import Random 

import subprocess


from torchvision.transforms.v2 import (
    Compose, ToImage, Resize, ToDtype, Normalize
)

import torch.nn.functional as F 
from core_utils import VisionTransformer

SEED = 42
torch.manual_seed(SEED)
BATCH = 64

device = (torch.accelerator.current_accelerator().type 
          if torch.accelerator.is_available() else "cpu")

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

MEAN = [0.485,0.456,0.406]
STD = [0.229,0.224,0.225]

transforms = Compose([
    Resize((224, 224)), ToImage(),
    ToDtype(torch.float32, scale=True),
    Normalize(MEAN, STD),
])


def create_data():
    train_data = torchvision.datasets.CIFAR10(
      root="./datasets", train=True, transform=transforms, download=True)
    test_data = torchvision.datasets.CIFAR10(
      root="./datasets", train=False, transform=transforms, download=True) 
    return (train_data,test_data)
    """
    n = range(512)
    train_n = Subset(train_data,n)
    test_n = Subset(test_data,n)
    return (train_n,test_n)"""

create_loader = lambda data,batch: DataLoader(data,batch,True)
# def create_loader(data,batch):
#     loader = DataLoader(data,batch,shuffle=True)
#     return loader

def test_metrics(model, loader, criterion, device="cuda"):
  model.eval()
  running_loss = 0.0
  running_acc = 0.0
  total = 0
  for images, labels in loader:
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

def create_model():
    vit_l_16 = VisionTransformer(L16Config)
    vit_l_16 = vit_l_16.to(device)
    vit_l_16.train()
    return vit_l_16

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
        
import time
from math import ceil
EPOCHS = 2
def train_single_epoch(rank,size,local_train_loader,local_test_loader,bsz):
    device = torch.device(f"cuda:{rank}")
    model = create_model()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=.9)
    criterion = torch.nn.CrossEntropyLoss()
    test_metrics(model,local_test_loader,criterion,device)
    model.train()
    num_batches = ceil(len(local_train_loader.dataset) // float(bsz))
    for EPOCH in range(EPOCHS):
        start = time.time()
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(local_train_loader):
            images = images.to(device)
            actual = labels.to(device)
            optimizer.zero_grad()
            predicted = model(images)
            loss = criterion(predicted,actual)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        stop = time.time()
        print(f"rank {rank} epoch loss: {epoch_loss / num_batches:.2f}")
        print(f"rank {rank} time : {(stop - start) / 3600:.5f} hours per epoch")
        test_metrics(model,local_test_loader,criterion,device)
        model.train()

def partition_dataset(rank,size):
    train_data,test_data = create_data()
    bsz = BATCH // size
    fracs = [1.0 / size for _ in range(size)]
    gen = torch.Generator().manual_seed(SEED)
    train_splits = random_split(train_data, fracs, generator=gen)
    test_splits = random_split(test_data, fracs, generator=gen)
    part_train = train_splits[rank]
    part_test = test_splits[rank]
    part_train_loader = create_loader(part_train, bsz)
    part_test_loader = create_loader(part_test, bsz)
    return (part_train_loader,part_test_loader,bsz)
 
def run(rank,size):
    torch.cuda.set_device(rank)
    gpu_name = torch.cuda.get_device_name(rank)
    local_train_loader, local_test_loader, local_batch = partition_dataset(rank,size)
    train_single_epoch(rank,size,local_train_loader,local_test_loader,local_batch)
    print(f"rank {rank} gpu {gpu_name}")
    
def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()    
    mp.set_start_method("spawn")
    mp.spawn(fn=init_process, args=(world_size,run), nprocs=world_size, join=True)
