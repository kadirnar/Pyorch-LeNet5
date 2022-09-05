import torch

from utils.dataset import MnistDataset


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device, num_epochs, batch_size):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.train_data_loader = MnistDataset(image_size=32, batch_size=64).__getitem__(train=True)
        self.test_data_loader = MnistDataset(image_size=32, batch_size=64).__getitem__(train=False)

    def train(self):
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if (i + 1) % 100 == 0:
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                            epoch + 1, self.num_epochs, i + 1, MnistDataset().__len__(train=True), loss.item()
                        )
                    )

        self.model = torch.save(self.model.state_dict(), "model.pt")

    def test(self):
        correct = 0
        total = 0
        for images, labels in self.test_data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / float(total)
        print("Accuracy of the network on the 10000 test images: {} %".format(acc))
