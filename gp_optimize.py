import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf_mixed
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from itertools import product


def get_next_hyperparameters(X, y, bounds, feature_type):
    # Introduce GP
    gp = SingleTaskGP(X, y)

    # Fit hyperparameters of the GP
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    
    # Bayesian Optimization
    # acquisition_fn = UpperConfidenceBound(gp, beta=3)
    acquisition_fn = ExpectedImprovement(gp, best_f=max(y))

    candidate, acq_value = optimize_acqf_mixed(
        acquisition_fn,
        fixed_features_list=generate_feature_list(bounds, feature_type),
        bounds=bounds, q=1, num_restarts=1, raw_samples=1
    )

    # Return next hyperparameters to try, and the associated acquisition value
    return candidate, acq_value


def generate_feature_list(bounds, feature_type):
    make_end_inclusive = np.array([0, 1])

    idx_discrete = np.argwhere(feature_type).squeeze()
    values = {i: np.arange(*(bounds[:, i] + make_end_inclusive))
              for i in idx_discrete}

    discrete_feature_list = []

    for values in product(*values.values()):
        feature_values = {idx_discrete[i]: v
                          for i, v in enumerate(values)}

        discrete_feature_list.append(feature_values)

    return discrete_feature_list
