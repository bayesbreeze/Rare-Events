# Importing the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
 

 
# Initializing the random seed
random_seed=1000

# Setting mean of the distributino
# to be at (0,0)
mean = np.array([0,0])
 
# Storing density function values for
# further analysis
pdf_list = []
 
val = 0.8

# Initializing the covariance matrix
cov = np.array([[1, val], [val, 1]])

# Generating a Gaussian bivariate distribution
# with given mean and covariance matrix


class MyDistr():
    def __init__(self):
        self.weights = np.ones(2)*0.25
        cov_vals = [0, 0]
        d = 3
        means = [np.array([d, 0]), np.array([-d, 0])]

        self.distrs = [multivariate_normal(cov = np.array([[1, cov_vals[i]], [cov_vals[i], 1]]), mean = means[i]) 
            for i in range(2)]
        
    def pdf(self, pos):
        val = 0
        for i in range(2):
            val += self.weights[i] * self.distrs[i].pdf(pos)
        return val
    

distr = MyDistr()

## ------------ do some plotting ------------------
plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize']=14,6
fig = plt.figure()

# Generating a meshgrid complacent with
# the 3-sigma boundary
mean_1, mean_2 = mean[0], mean[1]
sigma_1, sigma_2 = cov[0,0], cov[1,1]

x = np.linspace(-8*sigma_1, 8*sigma_1, num=100)
y = np.linspace(-8*sigma_2, 8*sigma_2, num=100)
X, Y = np.meshgrid(x,y)

# Generating the density function
# for each point in the meshgrid
pdf = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])

# Plotting the density function values
idx=0
key = 131+idx
ax = fig.add_subplot(key, projection = '3d')
ax.plot_surface(X, Y, pdf, cmap = 'viridis')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title(f'Covariance between x1 and x2 = {val}')
pdf_list.append(pdf)
ax.axes.zaxis.set_ticks([])
 
plt.tight_layout()
plt.show()
 
# Plotting contour plots
for idx, val in enumerate(pdf_list):
    plt.subplot(1,3,idx+1)
    plt.contourf(X, Y, val, cmap='viridis')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f'Covariance between x1 and x2 = ')
plt.tight_layout()
plt.show()

def logf(x):
    return torch.log()  \
             + torch.log(x[:,1]>3)
