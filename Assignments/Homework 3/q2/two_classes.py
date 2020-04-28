from datetime import datetime
import numpy as np
from torch import nn, optim
from torch.utils import data
from data import TwoClassCifar10
from model import ConvNet, LogisticRegression
from utils import train, test, lr_train, lr_test
from config import Config as config

print(f"{datetime.now().ctime()} - Start Loading Dataset...")
train_dataset = TwoClassCifar10(config.root, train=True)
test_dataset = TwoClassCifar10(config.root, train=False)
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
conv_net = ConvNet(config.input_channel, 2)
lr_model = LogisticRegression(config.cifar10_input_size)
conv_criterion = nn.CrossEntropyLoss()
lr_criterion = nn.BCEWithLogitsLoss()
conv_optimizer = optim.SGD(conv_net.parameters(),
                           config.lr,
                           momentum=config.momentum,
                           weight_decay=config.weight_decay)
lr_optimizer = optim.SGD(lr_model.parameters(),
                         config.lr,
                         momentum=config.momentum,
                         weight_decay=config.weight_decay)
conv_scheduler = optim.lr_scheduler.CosineAnnealingLR(conv_optimizer,
                                                      len(train_dataloader) *
                                                      config.epochs,
                                                      eta_min=config.eta_min)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(lr_optimizer,
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

conv_loss = []
conv_acc = []
for epoch in range(config.epochs):
    train(train_dataloader, conv_net, conv_criterion, conv_optimizer,
          conv_scheduler, config, epoch + 1)
    loss, acc = test(test_dataloader, conv_net, conv_criterion, epoch + 1)
    conv_loss.append(loss)
    conv_acc.append(acc)
print(f"{datetime.now().ctime()} - Finish Training")

lr_loss = []
lr_acc = []
for epoch in range(config.epochs):
    lr_train(train_dataloader, lr_model, lr_criterion, lr_optimizer,
             lr_scheduler, config, epoch + 1)
    loss, acc = lr_test(test_dataloader, lr_model, lr_criterion, epoch + 1)
    lr_loss.append(loss)
    lr_acc.append(acc)
print(f"{datetime.now().ctime()} - Finish Training")

plt.plot(conv_loss, label="ConvNet")
plt.plot(lr_loss, label="LR")
plt.title("Loss Comparison")
plt.legend()
plt.grid()
plt.show()

plt.plot(conv_acc, label="ConvNet")
plt.plot(lr_acc, label="LR")
plt.title("Acc Comparison")
plt.legend()
plt.grid()
plt.show()