import torch

def load_label_data(self, dataset_path):
    with open(os.path.join(dataset_path, "variants.txt"), "r") as f:
      return {index: label for index, label in enumerate(f.read().strip().split("\n"))}

def predict_label(image):
  dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/FGVCAirCraftImageDataset"
  index2label = load_label_data(dataset_path)

  device = torch.device("cuda")
  model = torch.load(model_path).to(device)
  image = torch.tensor([image], device=device)
  _, labels = torch.max(model(image), 1)
  return index2label[labels[0]]
