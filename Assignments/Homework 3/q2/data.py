from torch.utils import data
from torchvision import datasets, transforms

cifar_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])
fashionmnist_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])


class Cifar10(data.Dataset):
    def __init__(self, root, train=True):
        self.dataset = datasets.CIFAR10(root,
                                        train=train,
                                        transform=cifar_transforms,
                                        download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


class FashionMNIST(data.Dataset):
    def __init__(self, root, train=True):
        self.dataset = datasets.FashionMNIST(root,
                                             train=train,
                                             transform=fashionmnist_transforms,
                                             download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


class TwoClassCifar10(data.Dataset):
    def __init__(self, root, train=True):
        self.image_list = []
        self.label_list = []
        dataset = datasets.CIFAR10(root,
                                   train=train,
                                   transform=cifar_transforms,
                                   download=True)
        for image, label in dataset:
            if label in [3, 5]:
                self.image_list.append(image)
                if label == 3:
                    self.label_list.append(0)
                elif label == 5:
                    self.label_list.append(1)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.label_list[idx]
        return image, label