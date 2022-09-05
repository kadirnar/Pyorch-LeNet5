import torch
from torch import nn

from models.LeNet.model import LeNet5
from train import ModelTrainer


class Detector:
    def __init__(
        self, 
        model,
        num_epochs, 
        batch_size, 
        device,
        criterion,
        optimizer,
    ):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def train(self):
        self.model = LeNet5().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 5
        
        ModelTrainer(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
        ).train()

Detector(num_epochs=10, batch_size=64, device="cuda").train()
