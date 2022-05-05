import torch
import cv2
import json
import math
import copy
import time
import random
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
import torch
from  torch.cuda.amp import GradScaler, autocast
def train_model(model, optimizer, dataset, data_loader, num_epochs):
  since = time.time()
  # check if gpu is available
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  print('training with gpu...' if torch.cuda.is_available() else 'training with cpu...')
  model.to(device)
  # store best model weights
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 100000
  model.train()
  scaler = GradScaler()

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    torch.cuda.empty_cache()
    for phase in ['train', 'val']:
      running_loss = 0.0

      # iterate over data
      for images, targets in data_loader[phase]:
        # transfer data into requested input format of RetinaNet
        # transfer data to gpu if available
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # loss calculation
        with autocast():
            losses = model(images, targets)
            losses = sum(loss for loss in losses.values())

        # update the model only in the training phase
        # gradient back propagation + model parameters update
        if phase == 'train':
          optimizer.zero_grad()
          scaler.scale(losses.sum()).backward()
          scaler.step(optimizer)
        # accumulate loss values
        running_loss += losses.sum().item() * len(images)

      # average loss
      epoch_loss = running_loss / len(dataset[phase])
      print('{} loss: {:.4f}'.format(phase, epoch_loss))

      # select the best model in the validation phase
      # copy the model weights if loss is smaller
      if phase == 'val' and epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
    
    path = "Model{}.pt".format(epoch)
    torch.save(model.state_dict(), path)
    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
  print('Best validation loss: {:4f}'.format(best_loss))
  # load best model weights
  torch.save(best_model_wts, "model_wts.pt")
  model.load_state_dict(best_model_wts)
  return model
