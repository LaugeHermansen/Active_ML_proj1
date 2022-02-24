import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf_mixed
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement


def get_next_hyperparameters(X, y, bounds = None):
    # Introduce GP Surrogate Model
    gp = SingleTaskGP(X, y)

    # Fit hyperparameters of the GP
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    
    # Bayesian Optimization
    # Set default bounds if no bounds were provided
    if bounds is None:
        bounds = torch.stack([torch.zeros(len(X.T)), torch.ones(len(X.T))])

    # TODO: Implement HyBO
    # TODO: Implement support for arbitrary acquisition function
    # acquisition_fn = UpperConfidenceBound(gp, beta=3)
    acquisition_fn = ExpectedImprovement(gp, best_f=max(y))
    candidate, acq_value = optimize_acqf_mixed(
        acquisition_fn, bounds=bounds, q=1, num_restarts=1, raw_samples=1
    )

    # Return next hyperparameters to try, and the associated acquisition value
    return candidate, acq_value
