import torch, torchvision 
import os, sys 
import torch.distributed as dist
import torch.multiprocessing as mp 
from torch.utils.data import Dataset, DataLoader, Subset, random_split 
from random import Random 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import fully_shard, FSDPModule 

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



def test_metrics(model, loader, criterion, device="cuda"):
  model.eval()
  running_loss = 0.0
  running_acc = 0.0
  total = 0
  print("device:", device)
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

def create_model(device):
    print(f"model rank {dist.get_rank()} device {device}")
    vit_l_16 = VisionTransformer(L16Config)
    vit_l_16 = vit_l_16.to(device)
    return vit_l_16

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
        
import time
from tqdm import tqdm
EPOCHS = 2


def train(model, train_loader, test_loader, 
          optimizer, criterion, bsz, sampler=None):
    rank = dist.get_rank()
    size = dist.get_world_size()
    EPOCHS = 2
    model.train()
    print(f"rank {rank} local batch: {bsz}")
    for EPOCH in range(EPOCHS):
        if sampler:
           sampler.set_epoch(EPOCH)
        if rank==0:
            inner_pbar = tqdm(
             range(len(train_loader)), 
             colour="blue", desc="r0 Training Epoch"
            )
        start = time.time()
        epoch_loss = 0
        for images,labels in train_loader:
            images = images.to(rank)
            labels = labels.to(rank)
            predicted = model(images)
            loss = criterion(predicted,labels)
            epoch_loss += loss.item()
            if rank==0:
                inner_pbar.update(1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        stop = time.time()
        print(f"rank {rank} epoch loss: {epoch_loss / len(train_loader):.2f}")
        print(f"rank {rank} time : {(stop - start) / 3600:.5f} hours per epoch")
        test_metrics(model,test_loader,criterion,f"cuda:{rank}")
        model.train()

create_loader = lambda data,kwargs:DataLoader(data,**kwargs)
from torch.utils.data.distributed import DistributedSampler

def partition_dataset(rank,size,batch_size):
    train_data,test_data = create_data()
    sampler1 = DistributedSampler(train_data,rank=rank,num_replicas=size,shuffle=True)
    sampler2 = DistributedSampler(test_data,rank=rank,num_replicas=size)
    train_kwargs = {'batch_size': batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,'pin_memory': True,'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    train_loader = create_loader(train_data,train_kwargs)
    test_loader = create_loader(test_data,test_kwargs)
    return (train_loader,sampler1,test_loader,sampler2) 
    
def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.cuda.set_device(rank)
    fn(rank, size)
    dist.destroy_process_group()


def apply_fsdp(rank,size):
    vit_l_16 = VisionTransformer(L16Config)
    for vit_block in vit_l_16.vit_blocks:
        fully_shard(vit_block)
    fully_shard(vit_l_16)
    return vit_l_16
    
def train_fsdp(rank,size):
    fsdp_model = apply_fsdp(rank,size)
    assert isinstance(fsdp_model, VisionTransformer)
    assert isinstance(fsdp_model, FSDPModule)

    (train_loader, sampler1,
    test_loader, sampler2) = partition_dataset(rank,size,BATCH)
    optimizer = torch.optim.SGD(
                    fsdp_model.parameters(),lr=1e-3,momentum=.9)
    loss_fn = torch.nn.CrossEntropyLoss().to(rank)
    train(fsdp_model, train_loader, test_loader, 
          optimizer, loss_fn, BATCH, sampler1)
    """
    from looseversion import LooseVersion
    bf16_ready = (
        torch.version.cuda and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0" 
        and dist.is_nccl_available() 
        and torch.cuda.nccl.version() >= (2, 10)
    )
    """
    
    
if __name__ == "__main__":
    world_size = torch.cuda.device_count()    
    mp.set_start_method("spawn")
    mp.spawn(fn=init_process, args=(world_size,train_fsdp), nprocs=world_size, join=True)
