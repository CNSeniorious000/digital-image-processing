from torch import cuda, device

device = device("cuda:0" if cuda.is_available() else "cpu")
