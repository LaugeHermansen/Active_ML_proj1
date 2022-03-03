# %%
import numpy as np
from matplotlib import pyplot as plt
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from test import bounds as bounds_np, feature_type
#from botorch.optim import optimize_acqf_mixed
#from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement

def generate_plot(iteration, gif=False, nstart=10, hypers_path = "results/bootstrap_hyperparameters.npy", accs_path = "results/bootstrap_accuracies_val.npy"):
    """
    function to plot posterior mean and standard deviation for all 4x4 discrete dimensions of width and depth
    (If we extend our bounds we will need to do some manual modifications - sorry its not smarter...)

    Parameters
    ----------
    iteration:
        number of points the GP should be trained with and that are plotted
        Ensure that iteration is smaller than or equal to the total length of the hyperparameter and accuracies arrays
    gif:    
        boolean to determine color scheme (we can call generate plot in a loop to generate all necessary plots or a gif)
    n_start:
        default 10 randomly initialized starting points
        To let function know how many random points the GP starts with
    hypers_path: 
        String path to numpy file with hyperparameters
    accs_path:
        String path to numpy file with validation accuracies
    """
    # Redefine bounds from numpy array to torch tensor
    bounds = torch.tensor(bounds_np)

    hypers = np.load(hypers_path)
    accs = np.load(accs_path) 

    #Check that iterations is smaller than or equal to len of hyperparameters
    assert (iteration <= len(hypers)),"Iteration number cannot be larger than the number of hyperparameter points available!" 

    hypers = hypers[:iteration]
    accs = accs[:iteration]

    hypers = torch.tensor(hypers).reshape(-1, len(feature_type)) #X
    accs = torch.tensor(accs).reshape(-1, 1) #y

    #Define grid
    N = 100
    lrs = torch.linspace(bounds_np[0][0], bounds_np[1][0],N)
    lrs = lrs.repeat_interleave(N)
    wds = torch.linspace(bounds_np[0][1], bounds_np[1][1],N)
    wds = wds.repeat(N)

    # Normalize X
    hypers_norm = (hypers - bounds[0, :]) / (bounds[1, :] - bounds[0, :])

    # Normalize bounds
    bounds_norm = torch.tensor(np.ones_like(bounds))
    bounds_norm[0, :] = 0

    # Introduce GP
    gp = SingleTaskGP(hypers_norm, accs)

    # Fit hyperparameters of the GP
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    #EI = ExpectedImprovement(gp, best_f=max(accs))

    means = np.zeros((4,4,N,N))
    stds = np.zeros((4,4,N,N))
    #acqs = np.zeros((4,4,N,N))

    for depth in range(1,5):
        for width in range(1,5):
            post_mean = []
            space_params = torch.stack((lrs, wds, torch.ones(N**2)*depth, torch.ones(N**2)*width),1)
            space_params_norm = (space_params - bounds[0, :]) / (bounds[1, :] - bounds[0, :])
            post_mean = gp.posterior(space_params_norm).mean
            post_var = gp.posterior(space_params_norm).variance
            
            #ei = EI(space_params_norm).detach().numpy()
            
            post_mean = np.squeeze(post_mean.detach().numpy()).reshape((N,N))
            post_std = np.sqrt(np.squeeze(post_var.detach().numpy()).reshape((N,N)))

            means[depth-1, width-1] = post_mean
            stds[depth-1, width-1] = post_std
            #eis[depth-1, width-1] = ei

    def grid_plot(title, name, data):
        max = np.max(data.squeeze())
        min = np.min(data.squeeze())
        idx = 1
        f, axs = plt.subplots(4,4,figsize=(30,20))
        for width in range(1,5):
            for depth in range(1,5):
                plt.subplot(4, 4, idx)
                plt.contourf(np.linspace(bounds_np[0][0], bounds_np[1][0],N), np.linspace(bounds_np[0][1], bounds_np[1][1],N), data[depth-1, width-1], levels=30, vmin=min, vmax=max, cmap="rainbow")

                mask1 = hypers[:,2].detach().numpy() == depth
                mask2 = hypers[:,3].detach().numpy() == width
                if gif:
                    for p in hypers[mask1*mask2,:2].detach().numpy()[:-1]:
                        plt.plot(p[1],p[0],'k*') # weight decay on x, learning rate on y
                    if (mask1*mask2)[-1]:
                        last_point = hypers[mask1*mask2,:2].detach().numpy()[-1] # weight decay on x, learning rate on y
                        plt.plot(last_point[1],last_point[0],'w*',markersize=20)
                else:
                    mask3 = np.arange(iteration)[mask1*mask2]
                    for p, i in zip(hypers[mask1*mask2,:2].detach().numpy(),mask3):
                        plt.plot(p[1],p[0],'*', color=str(i/iteration)) # weight decay on x, learning rate on y
                
                plt.title(f'n_d: {depth}, n_w: {width}')
                idx += 1
                
        cax = plt.axes([0.95, 0.1, 0.01, 0.8])
        plt.colorbar(cax=cax)
        if gif:
            f.suptitle(f"Posterior {title}, iteration {iteration-nstart}", fontsize=45)
            plt.savefig(f"gif_plot/{name}{iteration-nstart}.png")
        else:
            f.suptitle(f"Posterior {title} of all {iteration} sampled datapoints", fontsize=45)
            plt.savefig(f"plot_of_{name}.png")

    grid_plot("Standard Deviation", "std", stds)
    grid_plot("Mean", "mean", means)
    
#Til Torben: bare brug denne kommando, gif generere vi imorgen
generate_plot(
    iteration=10, 
    gif=False, 
    nstart=10, 
    hypers_path = "results/bootstrap_hyperparameters.npy", 
    accs_path = "results/bootstrap_accuracies_val.npy"
)