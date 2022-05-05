import torch
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from model import obj_model
import sys
from dataset import CraterDataset
from train import train_model
from torch import nn

file_path = 'output.txt'
sys.stdout = open(file_path, "w")
model = obj_model()
# define data tranformation function
data_transforms = transforms.Compose([transforms.ToTensor()])
# define dataset and dataloader for training for debug set
dataset = {x: CraterDataset(x, data_transforms) for x in ['train', 'val']}
data_loader = {x: DataLoader(dataset[x], batch_size=8,
                             shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
for x in ['train', 'val']}
# transfer the model to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= nn.DataParallel(model)
model = model.to(device)
# define optimization algorithm
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# train model
model = train_model(model, optimizer, dataset, data_loader, num_epochs=10)
