import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm.auto import tqdm

#using matplotlib colourmaps to get colours

cmap = plt.get_cmap('gist_rainbow')
spectrum_resolution = 1000
# Need to define code that can take a distribution and sample from it

class true_distribution():
    def __init__(self,mus,sigmas,heights,SNR = 10):

        #check that all the lists are the same
        self.check_inputs(mus)
        self.check_inputs(sigmas)
        self.check_inputs(heights)

        assert (len(mus) == len(sigmas)) and (len(heights) == len(sigmas)), "Inputs need to be of same length"
        #defining a spectrum as a series of gaussians
        self.weights = np.zeros((spectrum_resolution),dtype = np.float32)
        self.x = np.linspace(0,1,spectrum_resolution,dtype = np.float32)

        for m,s,h in zip(mus,sigmas,heights):
            self.gaussian(m,s,h)
        # normalise weights
        self.weights /= np.sum(self.weights)
        assert np.sum(self.weights) >= 0.999999, f"Sum of weights =/= 1. Normalising likely went wrong somewhere. Expected 1, got {np.sum(self.weights)}"
        self.avg_color = self.get_colour()
        # assign SNR
        self.SNR = SNR

    def gaussian(self,mu,sigma,height):
        expo = -((self.x-mu)**2)/(2*sigma**2)
        self.weights = self.weights + height * np.exp(expo)

    def check_inputs(self,list_or_array):
        list_explainer = " is not a list or array. If only inputting one number put square brackets around it (e.g. 0.3 -> [0.3])"
        assert isinstance(list_or_array,list) or isinstance(list_or_array,np.ndarray), list_or_array.__name__ + list_explainer

    def plot_ideal_spectrum(self,show = True):
        plt.plot(self.x,self.weights,'k',alpha = 0.5)
        for i in range(spectrum_resolution - 1):
            plt.fill_between([self.x[i],self.x[i+1]],
                            [0,0],
                            [self.weights[i],self.weights[i+1]],
                            color = cmap(self.x[i]))
        add_avg_colour(plt.gca(),color = self.avg_color[:3])
        if show is True:
            plt.show()

    def get_colour(self):
        colors = cmap(self.x)
        
        return np.average(colors,axis = 0,weights = self.weights)
    def sample(self,n_samples = 1):
        pure_samples = np.random.choice(self.x,size = (n_samples),replace = True,p = self.weights)
        return(pure_samples)
def add_avg_colour(ax,color,corner = 1):
    y_lims = ax.get_ylim()
    x_lims = ax.get_xlim()
    
    size = 0.2
    if corner == 1:

        rect = patches.Rectangle([x_lims[corner] - size*x_lims[corner], y_lims[1] - size*y_lims[1]],
                                    width = size*x_lims[corner],height=size*y_lims[1],
                                    color = color)
    else:
        rect = patches.Rectangle([0, y_lims[1] - size*y_lims[1]],
                                    width = size*x_lims[1],height=size*y_lims[1],
                                    color = color)
    ax.add_patch(rect)
def bin_centers(bin_edges):
    return([(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges) - 1)])

def avg_colour(x,binned_samples):
    return(np.average(cmap(x),axis = 0,weights = binned_samples))

if __name__ == "__main__":
    test_dist = true_distribution([0.1,0.3,0.6],[0.05,0.01,0.025],[1,2,4],SNR = 1.5)
    n_samples = int(100000)
    samples = np.zeros(n_samples)

    samples = test_dist.sample(n_samples=n_samples)
        
    binned_values,binned_edges = np.histogram(samples,bins = spectrum_resolution,range = (0,1))
    binned_values = binned_values.astype(np.float32)
    binned_values/=np.sum(binned_values)
    
    test_dist.plot_ideal_spectrum(show = False)
    avg_colour_samples = avg_colour(test_dist.x,binned_samples=binned_values)
    add_avg_colour(plt.gca(),avg_colour_samples[:3],corner = 0)
    plt.plot(bin_centers(binned_edges),binned_values,color = 'k',linewidth = 1)
    
    
    plt.show()
    