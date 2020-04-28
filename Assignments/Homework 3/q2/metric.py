import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from data import TwoClassCifar10
from model import ConvNet, LogisticRegression
from config import Config as config

test_dataset = TwoClassCifar10(config.root, train=False)

conv_net = ConvNet(config.input_channel, 2)
lr_model = LogisticRegression(config.cifar10_input_size)
conv_net.load_state_dict(torch.load("model/2020428163925_0.719000.pth"))
lr_model.load_state_dict(torch.load("model/2020428163951_0.589000.pth"))

conv_preds = []
lr_preds = []
targets = []
with torch.no_grad():
    for image, label in test_dataset:
        image.unsqueeze_(0)
        conv_pred = conv_net(image)
        lr_pred = lr_model(image)
        conv_pred = torch.max(torch.softmax(conv_pred, dim=1),
                              dim=1)[0].squeeze()
        lr_pred = torch.sigmoid(lr_pred).squeeze()
        conv_preds.append(conv_pred.item())
        lr_preds.append(lr_pred.item())
        targets.append(label)

fpr, tpr, thresholds = metrics.roc_curve(targets, conv_preds)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr,
                                  tpr=tpr,
                                  roc_auc=roc_auc,
                                  estimator_name='ConvNet')
display.plot()
plt.show()

fpr, tpr, thresholds = metrics.roc_curve(targets, lr_preds)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr,
                                  tpr=tpr,
                                  roc_auc=roc_auc,
                                  estimator_name='LR')
display.plot()
plt.show()
