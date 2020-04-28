class Config:
    root = "./"
    dataset = "fashionmnist" # cifar10, fashionmnist
    model = "onelayer" # mlp, convnet, onelayer
    cifar10_input_size = 1024
    fashionmnist_input_size = 784
    input_channel = 1
    num_classes = 10
    batch_size = 64
    epochs = 100
    lr = batch_size / 256 * 0.1
    momentum = 0.9
    weight_decay = 5e-4
    eta_min = 1e-6
    print_interval = 100
    logistic_regression_iter = 1000