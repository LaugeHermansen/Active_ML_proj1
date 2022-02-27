from torchvision import datasets
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt

training_set = datasets.EMNIST(root='./data', split="byclass", train=True, download=True, transform= None)
test_set = datasets.EMNIST(root='./data', split="byclass", train=False, download=True, transform= None)

plt.plot(training_set)
plt.show()
