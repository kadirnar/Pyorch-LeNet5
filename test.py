import torch
from torch import nn

from model import LeNet5
from train import ModelTrainer


class ModelDetector:
    def __init__(self, num_epochs, batch_size, device):
        self.device = device
        self.model = LeNet5().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def train(self):
        ModelTrainer(
            self.model,
            self.criterion,
            self.optimizer,
            self.device,
            self.num_epochs,
        ).train()

    def test(self):
        ModelTrainer(
            self.model,
            self.criterion,
            self.optimizer,
            self.device,
            self.num_epochs,
        ).test()


ModelDetector(num_epochs=10, batch_size=64, device="cuda").train()
ModelDetector(num_epochs=10, batch_size=64, device="cpu").test()
