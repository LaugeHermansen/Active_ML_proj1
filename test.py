#%%
import botorch
import torch
from torchvision import datasets
import torch.utils.data as data_utils
import model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from gp_optimize import get_next_hyperparameters
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor()])

training_set = datasets.EMNIST(root="./data", split="byclass", train=True,  download=False, transform=transform)
test_set = datasets.EMNIST(root="./data", split="byclass", train=False,  download=False, transform=transform)
# NOTE: Temporarily only working with a subset of the dataset
training_set = data_utils.Subset(training_set, torch.arange(40000))
test_set = data_utils.Subset(test_set, torch.arange(40000))

train_dl = DataLoader(training_set, batch_size=200)
test_dl = DataLoader(test_set, batch_size=200)

#%%
accuracies = []
hypers = [np.array([0.001, 0.01])]
acquisitions = []

for i in range(40):
    # Instantiate and test model
    net = model.CNN_class(width=4, depth=4)
    model.train(net, train_dl, lr=hypers[-1][0], weight_decay=hypers[-1][1], n_epochs=2)
    accuracy = model.test(net, test_dl)

    # Track accuracy
    accuracies.append(accuracy)
    
    # Get next hyper parameters
    hyper_next, acqst = get_next_hyperparameters(
        torch.tensor(hypers).reshape(-1, 2),
        torch.tensor(accuracies).reshape(-1, 1),
        bounds = torch.tensor([[0.00001, 0.0001],
                               [0.035, 0.04]])
    )
    
    #Save next hyperparameters and associated acquisition
    hypers.append(hyper_next.detach().numpy().squeeze())
    acquisitions.append(acqst)
    print(i, accuracies[-1], hypers[-2][0], hypers[-2][1])
    if len(hypers) > 3:
        h = np.array(hypers)
        ax = plt.axes(projection='3d')

        ax.view_init(30, 120)
        ax.plot_trisurf(h[:-1, 0], h[:-1, 1], accuracies, cmap="viridis")
        ax.scatter(h[:-1, 0], h[:-1, 1], accuracies)
        plt.show()


#%%
%matplotlib widget
h = np.array(hypers)
ax = plt.axes(projection='3d')

ax.view_init(30, 120)
ax.plot_trisurf(h[:-1, 0], h[:-1, 1], accuracies, cmap="viridis")
ax.scatter(h[:-1, 0], h[:-1, 1], accuracies)
plt.show()
