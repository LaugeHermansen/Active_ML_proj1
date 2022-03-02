#%%
from re import I
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
from lauges_tqdm import tqdm


# learning rate, weight decay, width, depth
bounds = np.array([[0.0, 0.0, 1.0, 1.0],
                   [8.0, 8.0, 4.0, 4.0]])
feature_type = np.array([False, False, True, True])

# Batch size, data loading workers
batch_size, data_workers = 200, 12


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])

    training_set = datasets.EMNIST(root="./data", split="byclass", train=True,  download=True, transform=transform)
    test_set = datasets.EMNIST(root="./data", split="byclass", train=False,  download=True, transform=transform)

    _, train_indices = train_test_split(np.arange(len(training_set)), test_size=0.4, stratify=training_set.targets)
    training_set = data_utils.Subset(training_set, train_indices)

    val_indices, test_indices = train_test_split(np.arange(len(test_set)), test_size=0.5, stratify=test_set.targets)
    validation_set = data_utils.Subset(test_set, val_indices)
    test_set = data_utils.Subset(test_set, test_indices)

    train_dl = DataLoader(training_set, batch_size=batch_size, num_workers=data_workers)
    validation_dl = DataLoader(validation_set, batch_size=batch_size, num_workers=data_workers)
    test_dl = DataLoader(test_set, batch_size=batch_size, num_workers=data_workers)

    return train_dl, validation_dl, test_dl


def get_random_hypers():
    return [np.random.randint(bounds[0, i], bounds[1, i] + 1) if feature else np.random.uniform(*bounds[:, i]) for
            i, feature in enumerate(feature_type)]


def get_random_hypers_acq(X, y, bounds, feature_type):
    return get_random_hypers(), np.Infinity


def get_bayesian_optimization_hypers_acq(X, y, bounds, feature_type):
    # Get next hyper parameters
    hypers, acqst = get_next_hyperparameters(
        torch.tensor(X).reshape(-1, len(feature_type)),
        torch.tensor(y).reshape(-1, 1),
        bounds=torch.tensor(bounds),
        feature_type=feature_type
    )

    # Return hypers and associated acquisition
    return hypers.detach().numpy().squeeze(), acqst


def train_test(log_learning_rate, log_weight_decay, width, depth):
    net = model.CNN_class(width=width, depth=depth)
    model.train(net, train_dl, lr=10**(-log_learning_rate), weight_decay=10**(-log_weight_decay), n_epochs=2)
    accuracy_val = model.test(net, validation_dl)
    accuracy_test = model.test(net, test_dl)
    
    return accuracy_val, accuracy_test, net


def save_results(name, new_hypers, new_accuracy_val, new_accuracy_test, new_acquisition, model, hypers, accuracies_val, accuracies_test, acquisitions, start_iteration = 0):
        # NOTE: Other than IO operations, this function is pure

        # If the new accuracy is better than best known, update model
        if len(accuracies_val[start_iteration:]) == 1 or new_accuracy_val > max(accuracies_val[start_iteration:]):
            torch.save(model.state_dict(), f"results/{name}_best_model.pt")

        # Append new hypers and accuracies to existing lists
        hypers = hypers + [new_hypers]
        accuracies_val = accuracies_val + [new_accuracy_val]
        accuracies_test = accuracies_test + [new_accuracy_test]
        acquisitions = acquisitions + [new_acquisition]

        # Save accuracies and hyper parameters
        np.save(f"results/{name}_hyperparameters", hypers)
        np.save(f"results/{name}_accuracies_val", accuracies_val)
        np.save(f"results/{name}_accuracies_test", accuracies_test)
        np.save(f"results/{name}_acquisitions", acquisitions)
        
        # Return now arrays
        return hypers, accuracies_val, accuracies_test, acquisitions


def optimize_model(name, new_hypers_fn, iterations, bootstrap_model_name = None):
    # Set random seeds
    np.random.seed(9327642)
    torch.manual_seed(830994)
    torch.use_deterministic_algorithms(True)


    # If bootstrapped points should be used, load them
    if bootstrap_model_name:
        hypers = list(np.load(f"results/{bootstrap_model_name}_hyperparameters.npy"))
        accuracies_val = list(np.load(f"results/{bootstrap_model_name}_accuracies_val.npy"))
        accuracies_test = list(np.load(f"results/{bootstrap_model_name}_accuracies_test.npy"))
        acquisitions = list(np.load(f"results/{bootstrap_model_name}_acquisitions.npy"))

        new_hypers, new_acquisition = new_hypers_fn(
            hypers, accuracies_val,
            bounds, feature_type
        )
    # Else, initialize with a random hyper parameter
    else:
        hypers = []
        accuracies_val = []
        accuracies_test = []
        acquisitions = []

        new_hypers, new_acquisition = get_random_hypers_acq(
            hypers, accuracies_val, 
            bounds, feature_type
        )


    # Track starting iteration
    start_iteration = len(hypers) - 1
    
    
    # Optimization loop
    print(f"Optimizing via {name}...")
    
    for _ in tqdm(range(iterations), n_child_layers=3):
        # WARN: Ensure order of parameters are same as function
        new_accuracy_val, new_accuracy_test, model = train_test(*new_hypers)

        # Save results
        hypers, accuracies_val, accuracies_test, acquisitions \
            = save_results(
                name, 
                new_hypers, new_accuracy_val, new_accuracy_test, new_acquisition, model, 
                hypers, accuracies_val, accuracies_test, acquisitions, start_iteration
            )
            
        # Get next hypers
        new_hypers, new_acquisition = new_hypers_fn(
            hypers, accuracies_val,
            bounds, feature_type
        )
        
        
def create_bootstrap_samples(name = "bootstrap", iterations = 10):
    # Set random seeds
    np.random.seed(7231487)
    torch.manual_seed(132082)
    torch.use_deterministic_algorithms(True)
    
    # Generate and save bootstrap file
    optimize_model(
        name, 
        get_random_hypers_acq,
        iterations
    )


# Load data globally
train_dl, validation_dl, test_dl = load_data()



if __name__ == '__main__':

    # create_bootstrap_samples(
    #     name="bootstrap", 
    #     iterations=10
    # )

    optimize_model(
        name="bayesian_optimization", 
        new_hypers_fn=get_bayesian_optimization_hypers_acq,
        iterations=400,
        bootstrap_model_name="bootstrap"
    )
    
    optimize_model(
        name="random_optimizer", 
        new_hypers_fn=get_random_hypers_acq,
        iterations=400,
        bootstrap_model_name="bootstrap"
    )