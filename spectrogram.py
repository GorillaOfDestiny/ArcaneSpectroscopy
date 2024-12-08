import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#using matplotlib colourmaps to get colours

cmap = plt.get_cmap('gist_rainbow')
spectrum_resolution = 1000
# Need to define code that can take a distribution and sample from it

class true_distribution():
    def __init__(self,mus,sigmas,heights):

        #check that all the lists are the same
        self.check_inputs(mus)
        self.check_inputs(sigmas)
        self.check_inputs(heights)

        assert (len(mus) == len(sigmas)) and (len(heights) == len(sigmas)), "Inputs need to be of same length"
        #defining a spectrum as a series of gaussians
        self.weights = np.zeros((spectrum_resolution))
        self.x = np.linspace(0,1,spectrum_resolution)

        for m,s,h in zip(mus,sigmas,heights):
            self.gaussian(m,s,h)
        # normalise weights
        self.weights /= np.sum(self.weights)
        assert np.sum(self.weights) >= 0.999999, f"Sum of weights =/= 1. Normalising likely went wrong somewhere. Expected 1, got {np.sum(self.weights)}"
        self.avg_color = self.get_colour()


    def gaussian(self,mu,sigma,height):
        expo = -((self.x-mu)**2)/(2*sigma**2)
        self.weights = self.weights + height * np.exp(expo)

    def check_inputs(self,list_or_array):
        list_explainer = " is not a list or array. If only inputting one number put square brackets around it (e.g. 0.3 -> [0.3])"
        assert isinstance(list_or_array,list) or isinstance(list_or_array,np.ndarray), list_or_array.__name__ + list_explainer

    def plot_ideal_spectrum(self):
        plt.plot(self.x,self.weights,color = 'k')
        for i in range(spectrum_resolution - 1):
            plt.fill_between([self.x[i],self.x[i+1]],
                            [0,0],
                            [self.weights[i],self.weights[i+1]],
                            color = cmap(self.x[i]))
        y_lims = plt.gca().get_ylim()
        x_lims = plt.gca().get_xlim()
        
        size = 0.2

        rect = patches.Rectangle([x_lims[1] - size*x_lims[1], y_lims[1] - size*y_lims[1]],
                                 width = size*x_lims[1],height=size*y_lims[1],
                                 color = self.avg_color[:3])
        plt.gca().add_patch(rect)
        plt.show()

    def get_colour(self):
        colors = cmap(self.x)
        
        return np.average(colors,axis = 0,weights = self.weights)

if __name__ == "__main__":
    test_dist = true_distribution([0.1,0.3,0.6],[0.05,0.01,0.025],[1,2,4])
    test_dist.plot_ideal_spectrum()