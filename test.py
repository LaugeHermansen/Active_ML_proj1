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
from sklearn.model_selection import train_test_split

def get_randoms(bounds, feature_type):
    return [np.random.randint(bounds[0, i], bounds[1, i] + 1) if feature else np.random.uniform(*bounds[:, i]) for
            i, feature in enumerate(feature_type)]

transform = transforms.Compose([transforms.ToTensor()])

training_set = datasets.EMNIST(root="./data", split="byclass", train=True,  download=True, transform=transform)
test_set = datasets.EMNIST(root="./data", split="byclass", train=False,  download=True, transform=transform)

_, train_indices = train_test_split(np.arange(len(training_set)), test_size=0.1, stratify=training_set.targets)
training_set = data_utils.Subset(training_set, train_indices)

val_indices, test_indices = train_test_split(np.arange(len(test_set)), test_size=0.5, stratify=test_set.targets)
validation_set = data_utils.Subset(test_set, val_indices)
test_set = data_utils.Subset(test_set, test_indices)

train_dl = DataLoader(training_set, batch_size=200)
validation_dl = DataLoader(validation_set, batch_size=200)
test_dl = DataLoader(test_set, batch_size=200)

#%%
accuracies_val = []
accuracies_test = []
accuracies_val_random = []
accuracies_test_random = []
#bounds = torch.tensor([[1e-6, 1e-6, 1, 1],
#                       [1/2, 1/2, 4, 4]])
bounds = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                       [4.0, 4.0, 4.0, 4.0]])
feature_type = [False, False, True, True] # lr, weight decay, width, depth
hypers = [np.array(get_randoms(bounds, feature_type))]
hypers_random = [np.array(get_randoms(bounds, feature_type))]

acquisitions = []

current_best = 0
current_best_random = 0

for i in range(5000):
    # Instantiate and test model
    net = model.CNN_class(width=hypers[-1][2], depth=hypers[-1][3])
    model.train(net, train_dl, lr=10**(-hypers[-1][0]), weight_decay=10**(-hypers[-1][1]), n_epochs=2)
    accuracy_val = model.test(net, validation_dl)
    accuracy_test = model.test(net, test_dl)

    if accuracy_val > current_best:
        current_best = accuracy_val
        torch.save(net, "best_model")

    # Track accuracy
    accuracies_val.append(accuracy_val)
    accuracies_test.append(accuracy_test)

    np.save("hyperparameters", hypers)
    np.save("accuracies_val", accuracies_val)
    np.save("accuracies_test", accuracies_test)

    # Get next hyper parameters
    hyper_next, acqst = get_next_hyperparameters(
        torch.tensor(hypers).reshape(-1, len(feature_type)),
        torch.tensor(accuracies_val).reshape(-1, 1),
        bounds=bounds,
        feature_type=feature_type
    )

    #Save next hyperparameters and associated acquisition
    hypers.append(hyper_next.detach().numpy().squeeze())
    acquisitions.append(acqst)
    print(i, accuracies_val[-1])

    # Random model:
    net = model.CNN_class(width=hypers_random[-1][2], depth=hypers_random[-1][3])
    model.train(net, train_dl, lr=10**(-hypers_random[-1][0]), weight_decay=10**(-hypers_random[-1][1]), n_epochs=2)
    accuracy_val = model.test(net, validation_dl)
    accuracy_test = model.test(net, test_dl)

    if accuracy_val > current_best_random:
        current_best_random = accuracy_val
        torch.save(net, "best_model_random")

    # Track accuracy
    accuracies_val_random.append(accuracy_val)
    accuracies_test_random.append(accuracy_test)

    np.save("hyperparameters_random", hypers_random)
    np.save("accuracies_val_random", accuracies_val_random)
    np.save("accuracies_test_random", accuracies_test_random)

    hypers_random.append(np.array(get_randoms(bounds, feature_type)))



#%%
#%matplotlib widget

