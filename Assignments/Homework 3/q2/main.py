from datetime import datetime
from torch import nn, optim
from torch.utils import data
from data import Cifar10, FashionMNIST
from model import MLP, ConvNet, OneLayer
from utils import train, test
from config import Config as config

print(f"{datetime.now().ctime()} - Start Loading Dataset...")
if config.dataset == "cifar10":
    train_dataset = Cifar10(config.root, train=True)
    test_dataset = Cifar10(config.root, train=False)
elif config.dataset == "fashionmnist":
    train_dataset = FashionMNIST(config.root, train=True)
    test_dataset = FashionMNIST(config.root, train=False)
train_dataloader = data.DataLoader(train_dataset,
                                   config.batch_size,
                                   shuffle=True,
                                   num_workers=2)
test_dataloader = data.DataLoader(test_dataset,
                                  config.batch_size,
                                  shuffle=False,
                                  num_workers=2)
print(f"{datetime.now().ctime()} - Finish Loading Dataset")

print(
    f"{datetime.now().ctime()} - Start Creating Net, Criterion, Optimizer and Scheduler..."
)
if config.model == "mlp":
    net = MLP(config.cifar10_input_size, config.num_classes)
elif config.model == "convnet":
    net = ConvNet(config.input_channel, config.num_classes)
elif config.model == "onelayer":
    net = OneLayer(config.fashionmnist_input_size, config.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      config.lr,
                      momentum=config.momentum,
                      weight_decay=config.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 len(train_dataloader) *
                                                 config.epochs,
                                                 eta_min=config.eta_min)
print(
    f"{datetime.now().ctime()} - Finish Creating Net, Criterion, Optimizer and Scheduler"
)

print(f"{datetime.now().ctime()} - Start Training...")
print(
    f"Traing dataset: {len(train_dataset)}, iteration: {len(train_dataloader)}"
)
print(
    f"Testing dataset: {len(test_dataset)}, iteration: {len(test_dataloader)}")
print(
    f"Epochs: {config.epochs}, Batch Size: {config.batch_size}, LR:{config.lr}",
    end='\n\n')

for epoch in range(config.epochs):
    train(train_dataloader, net, criterion, optimizer, scheduler, config,
          epoch + 1)
    test(test_dataloader, net, criterion, epoch + 1)

print(f"{datetime.now().ctime()} - Finish Training")