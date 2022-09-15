import torch

from utils.dataset import MnistDataset
from models.LeNet.model import LeNet5

class ModelTrainer:
    def config(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LeNet5().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_epochs = 5
        self.batch_size = 64
        

    def train(self):
        train_data_loader = MnistDataset().__getitem__(train=True)
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_data_loader):
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


        test_data_loader = MnistDataset().__getitem__(train=False)
        correct = 0
        total = 0
        for images, labels in test_data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / float(total)
        print("Accuracy of the network on the 10000 test images: {} %".format(acc))

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.config()
    trainer.train()
