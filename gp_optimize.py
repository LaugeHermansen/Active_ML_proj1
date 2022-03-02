import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf_mixed
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from itertools import product

def normalize(A, bounds):
    return (A - bounds[0, :]) / (bounds[1, :] - bounds[0, :])


def get_next_hyperparameters(X, y, bounds, feature_type):
    # Normalize X
    X_norm = normalize(X, bounds)
    
    # Normalize bounds
    bounds_norm = torch.tensor(np.ones_like(bounds))
    bounds_norm[0, :] = 0
    
    # Get normalized discrete feature list
    feature_list_norm = generate_feature_list(bounds.detach().numpy(), feature_type)


    # Introduce GP
    gp = SingleTaskGP(X_norm, y)

    # Fit hyperparameters of the GP
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    

    # Bayesian Optimization
    acquisition_fn = ExpectedImprovement(gp, best_f=max(y))

    candidate, acq_value = optimize_acqf_mixed(
        acquisition_fn,
        fixed_features_list=feature_list_norm,
        bounds=bounds_norm, q=1, num_restarts=1, raw_samples=1
    )


    # Unnormalize
    candidate = candidate * (bounds[1, :] - bounds[0, :]) + bounds[0, :]
    candidate[:, feature_type] = torch.round(candidate[:, feature_type])
    

    # Return next hyperparameters to try, and the associated acquisition value
    return candidate, acq_value


def generate_feature_list(bounds, feature_type):
    make_end_inclusive = np.array([0, 1])

    idx_discrete = np.argwhere(feature_type).squeeze()
    values = {i: normalize(np.arange(*(bounds[:, i] + make_end_inclusive), 
                                     dtype=np.float), 
                           bounds[:, i, None])
              for i in idx_discrete}

    discrete_feature_list = []

    for values in product(*values.values()):
        feature_values = {idx_discrete[i]: v
                          for i, v in enumerate(values)}

        discrete_feature_list.append(feature_values)


    return discrete_feature_list
