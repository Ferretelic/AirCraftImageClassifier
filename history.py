import matplotlib.pyplot as plt
import pickle
import os

def plot_history(history, history_path):
  with open(os.path.join(history_path, "history.pkl"), "wb") as f:
    pickle.dump(history, f)

  train_loss = history["train_loss"]
  validation_loss = history["validation_loss"]

  plt.figure()
  plt.plot(train_loss)
  plt.plot(validation_loss)
  plt.title("Loss")
  plt.savefig(os.path.join(history_path, "loss.png"))

  if "train_accuracy" in history.keys():
    train_accuracy = history["train_accuracy"]
    validation_accuracy = history["validation_accuracy"]

    plt.figure()
    plt.plot(train_accuracy)
    plt.plot(validation_accuracy)
    plt.title("Accuracy")
    plt.savefig(os.path.join(history_path, "accuracy.png"))