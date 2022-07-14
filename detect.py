from model import LeNet5
from train import ModelTrainer
from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
ModelTrainer(model, criterion, optimizer, device, num_epochs=10, batch_size=64).train()
ModelTrainer(model, criterion, optimizer, device, num_epochs=10, batch_size=64).test()