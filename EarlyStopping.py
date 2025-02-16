import torch
import os

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path="best_model.pth"):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            delta (float): Minimum change in loss to be considered as improvement.
            path (str): File path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        """
        Checks if validation loss has improved. If not, increases counter. 
        Stops training when counter reaches patience.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # torch.save(model.state_dict(), self.path)  # Save best model
            os.makedirs(self.path, exist_ok=True)
            torch.save(model.state_dict(), f'{self.path}/classification.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
