import os

import torch
import torchvision
import pyprind

from aircraft import create_train_validation_dataloaders
from history import plot_history

def train(model, optimizer, scheduler, criterion, epochs, train_dataloader, validation_dataloader, model_path, device):
  train_accuracy = []
  validation_accuracy = []
  train_loss = []
  validation_loss = []
  best_loss = float("inf")

  for epoch in range(epochs):
    running_loss = 0.0
    running_corrects = 0
    bar = pyprind.ProgBar(len(train_dataloader), title="Epoch: {:3d}".format(epoch + 1))

    for images, labels in train_dataloader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()

      outputs = model(images)
      _, predictions = torch.max(outputs, 1)

      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * images.size(0)
      running_corrects += torch.sum(predictions == labels).item()
      bar.update()

    train_loss.append(running_loss / float(len(train_dataloader.dataset)))
    train_accuracy.append(running_corrects / float(len(train_dataloader.dataset)))

    with torch.no_grad():
      evaluating_loss = 0
      evaluating_corrects = 0
      for images, labels in validation_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        evaluating_loss += loss.item() * images.size(0)
        evaluating_corrects += torch.sum(predictions == labels)

      validation_loss.append(evaluating_loss  / float(len(validation_dataloader.dataset)))
      validation_accuracy.append(evaluating_corrects / float(len(validation_dataloader.dataset)))

      if validation_loss[-1] < best_loss or validation_accuracy[-1] > best_accuacy:
        torch.save(model, os.path.join(model_path, "model_{}.pth".format(epoch + 1)))
        best_loss = validation_loss[-1]
        best_accuracy = validation_accuracy[-1]


    print("Epoch: {:3d}".format(epoch + 1))
    print("Train Accuracy: {:2.3f}, Validation Accuracy: {:2.3f}".format(train_accuracy[-1], validation_accuracy[-1]))
    print("Train Loss: {:2.3f}, Validation Loss: {:2.3f}".format(train_loss[-1], validation_loss[-1]))
    print("")
    scheduler.step()

  history = {"train_loss": train_loss, "validation_loss": validation_loss, "best_loss": best_loss}

  return history

device = torch.device("cuda")
model = torchvision.models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, 100)

model.to(device)
criterion = torch.nn.CrossEntropyLoss()
learning_rate = 0.01
scheduler_gamma = 0.65
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_lambda = lambda epoch: scheduler_gamma ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
epochs = 10
model_path = "./models"

dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/FGVCAirCraftImageDataset"
image_size = (224, 224)
batch_size = 64
train_dataloader, validation_dataloader = create_train_validation_dataloaders(dataset_path, image_size, batch_size)

history = train(model, optimizer, scheduler, criterion, epochs, train_dataloader, validation_dataloader, model_path, device)
best_loss = history["best_loss"]
plot_history(history, history_path)