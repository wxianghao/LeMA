import torchvision

root_path = "./data"
trainset = torchvision.datasets.MNIST(root_path, train=True, download=True)
testset = torchvision.datasets.MNIST(root_path, train=False)
