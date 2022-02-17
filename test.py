import botorch
import torch
from torchvision import datasets
import model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])

training_set = datasets.EMNIST(root="./data", split="byclass", train=True,  download=False, transform=transform)
test_set = datasets.EMNIST(root="./data", split="byclass", train=False,  download=False, transform=transform)

train_dl = DataLoader(training_set, batch_size=50)
test_dl = DataLoader(test_set, batch_size=50)

# test :
net = model.CNN_class(width=4, depth=4)

model.train(net, train_dl, lr=0.001, weight_decay=0.01, n_epochs=2)

print(model.test(net, test_dl))
