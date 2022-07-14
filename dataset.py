from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class MnistDataset(Dataset):
    def __init__(
        self,
        image_size: int = 32,
        batch_size: int = 64,
    ):
        self.image_size = image_size
        self.batch_size = batch_size

    def __getitem__(self, train=True):
        if train:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(self.image_size, self.image_size)),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            train_dataset = datasets.MNIST(root="dataset", train=True, transform=train_transform, download=True)
            data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        else:
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(self.image_size, self.image_size)),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            test_dataset = datasets.MNIST(root="dataset", train=False, transform=test_transform, download=True)
            data_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return data_loader

    def __len__(self, train=True):
        if train:
            return len(self.__getitem__(train=True))
        else:
            return len(self.__getitem__(train=False))
