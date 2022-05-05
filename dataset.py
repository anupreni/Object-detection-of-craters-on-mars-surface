import torch
import cv2
import json
import random
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision


class CraterDataset(Dataset):
    def __init__(self, phase, data_transforms):
    self.transform = data_transforms
    # load image IDs of training set or validation set
    self.img_ids = json.load(open('./ids_demo.json'))[phase]
    # load all crater bounding boxes
    self.crater_boxes = json.load(open('./gt_public.json'))


  def __getitem__(self, idx):
    img_id = self.img_ids[idx]
    # load the image
    img = cv2.imread("./images/{}.png".format(img_id))
    # get bounding boxes and transfer it
    # from [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax]
    crater_boxes = self.crater_boxes[img_id]
    num_craters = len(crater_boxes)
    boxes = []
    for box in crater_boxes:
        xmin = box[0]
        xmax = box[0] + box[2]
        ymin = box[1]
        ymax = box[1] + box[3]
        boxes.append([xmin, ymin, xmax, ymax])

    # tranfser to requested data type
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # the class of all boxes is the same. 0: background 1: crater
    labels = torch.ones((num_craters,), dtype=torch.int64)
    # wrap up as a dictionary
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    # data transform
    if self.transform is not None:
        img = self.transform(img)

    return img, target


  def __len__(self):
    return len(self.img_ids)
