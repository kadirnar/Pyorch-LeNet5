import torch
from torch import nn
from model import LeNet5
from dataset import MnistDataset


images, labels = MnistDataset(image_size=32, batch_size=64).__getitem__(train=True)

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device, num_epochs, batch_size):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def train(self, train_loader):
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if (i+1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, i+1, len(train_loader)//self.batch_size, loss.item()))

    def test(self, test_loader):
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / float(total)
        print('Accuracy of the network on the 10000 test images: {} %'.format(acc))