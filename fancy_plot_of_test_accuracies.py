#%%
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracies(accuracies: np.ndarray, legend, ylabel, xlabel, plot_title, label_size = None, title_size = None):
        
    """
    function to plot accuracies and with current accuracies to compare optimization models

    Parameters
    ----------
    accuracies :
        numpy array of accuracies with shape (N,M) - N data points (iterations) and M optimization algorithms (probably just BO with GP vs random)
    legend :    
        the names of the optimizaion models
    ylabel :    
        you know
    xlabel :    
            -----
    plot_title :
            -----
    label_size :
        font size of x_label and y_label
    title_size :
        font size of title
    """
    
    N,M = accuracies.shape
    if len(legend) != M:  raise ValueError("The length of legend doesn't correspond with the number og lines to plot")
    
    current_max_accuracies = accuracies.copy()
    for i in range(1,N):
        current_max_accuracies[i] = np.max([current_max_accuracies[i-1],accuracies[i]],axis = 0)
    for i in range(M):
        ax = plt.plot(current_max_accuracies[:,i],'-',label = legend[i])
        plt.plot(accuracies[:,i],'--',alpha = 0.8, linewidth = 0.6, color = ax[0].get_color(), label = "")#f"Current max of {legend[i].lower()}")
    
    plt.legend()
    plt.ylabel(ylabel, fontsize = label_size)
    plt.xlabel(xlabel, fontsize = label_size)
    plt.title(plot_title, fontsize = title_size)
    plt.show()


if __name__ == "__main__":
    N = 200
    accuracy1 = np.log(np.arange(1,N+1))**3 + np.random.rand(N)*40
    accuracy2 = np.arange(N)*0.1 + np.random.rand(N)*40
    accuracies = np.vstack((accuracy1,accuracy2)).T
    plot_accuracies(accuracies,["Hej","Lauge"],"Validation acc in %", "Iteration", "Validation acc vs Iteration")



