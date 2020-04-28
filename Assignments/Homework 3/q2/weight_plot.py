import torch
from model import OneLayer
from config import Config as config
import matplotlib.pyplot as plt

label_dict = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}
model = OneLayer(config.fashionmnist_input_size, config.num_classes)
model.load_state_dict(torch.load("model/202042823737_0.845300.pth"))

weight = model.linear.weight.data.numpy()

for idx in range(10):
    vec = weight[idx]
    plt.imshow(vec.reshape(28, 28))
    plt.title(f"Label {idx}: {label_dict[idx]}")
    plt.show()