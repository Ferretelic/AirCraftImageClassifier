import os

import torch
import torchvision
from PIL import Image

class AirCraftDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, image_size, mode):
    self.mode = mode
    self.compose_transforms(image_size)
    self.load_label_data(dataset_path)

    if mode == "train":
      file_name = "images_variant_train.txt"
    elif mode == "validation":
      file_name = "images_variant_val.txt"
    elif mode == "test":
      file_name = "images_variant_test.txt"

    image_paths = []
    image_labels = []
    with open(os.path.join(dataset_path, file_name), "r") as f:
      for image_data in f:
        image_name, image_label = image_data.strip().split(" ", 1)
        image_paths.append(os.path.join(dataset_path, "images", "{}.jpg".format(image_name)))
        image_labels.append(self.label2index[image_label])

    self.images = image_paths
    self.labels = image_labels

  def __getitem__(self, index):
    image_path = self.images[index]
    image = Image.open(image_path).convert("RGB")
    label = torch.tensor(self.labels[index], dtype=torch.long)

    if self.mode == "train":
      image = self.train_transform(image)
    elif self.mode == "validation":
      image = self.validation_transform(image)
    elif self.mode == "test":
      image = self.test_transform(image)

    return image, label

  def __len__(self):
    return len(self.images)

  def compose_transforms(self, image_size):
    self.train_transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize(image_size),
      torchvision.transforms.RandomRotation(15),
      torchvision.transforms.RandomCrop(image_size, padding=24),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    self.validation_transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize(image_size),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    self.test_transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize(image_size),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

  def load_label_data(self, dataset_path):
    with open(os.path.join(dataset_path, "variants.txt"), "r") as f:
      self.label2index = {label: index for index, label in enumerate(f.read().strip().split("\n"))}

def create_train_validation_dataloaders(dataset_path, image_size, batch_size):
  train_dataset = AirCraftDataset(dataset_path, image_size, "train")
  validation_dataset = AirCraftDataset(dataset_path, image_size, "validation")

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
  validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)

  return train_dataloader, validation_dataloader

def transform_image(image):
  transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize((224, 224)),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  return transform(image)
