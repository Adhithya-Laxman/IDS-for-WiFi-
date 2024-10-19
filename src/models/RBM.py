
import numpy as np
import torch
import random
from tqdm import trange
'''
m,v in this file: 

In the context of your RBM class implementation, the variables m and v are used as part of the Adam optimization algorithm. They help maintain the first and second moment estimates of the gradients during training. Here's what they represent:

m: This is the first moment estimate, which is the exponential moving average of the gradients. It essentially captures the mean of the gradients over time. The update formula for m is:

m_t = β1 * m_(t-1) + (1 - β1) * g_t

where:

m_t is the current first moment estimate,
m_(t-1) is the previous first moment estimate,
g_t is the current gradient,
β1 is the decay rate for the first moment (usually set to 0.9).
v: This is the second moment estimate, which is the exponential moving average of the squared gradients. It captures the variance of the gradients. The update formula for v is:

v_t = β2 * v_(t-1) + (1 - β2) * g_t^2

where:

v_t is the current second moment estimate,
v_(t-1) is the previous second moment estimate,
g_t^2 is the square of the current gradient,
β2 is the decay rate for the second moment (usually set to 0.999).
Initialization in Your Code
In your RBM class, m and v are initialized as lists with three elements, representing the three parameters being optimized: the weights W, the visible biases vb, and the hidden biases hb.

python
Copy code
self.m = [0, 0, 0]  # First moment estimates for W, vb, hb
self.v = [0, 0, 0]  # Second moment estimates for W, vb, hb
Usage in Adam Optimization
During the update phase, when you call the adam method, it calculates the moment estimates for each parameter and adjusts the learning rate accordingly. Here's the part of your code where this happens:

python
Copy code
if self.optimizer == 'adam':
    dW = self.adam(dW, epoch, 0)
    dvb = self.adam(dvb, epoch, 1)
    dhb = self.adam(dhb, epoch, 2)
For each parameter (W, vb, hb), adam updates the corresponding m and v values, allowing the Adam optimizer to adaptively adjust the learning rates based on the history of each parameter's gradients. This results in more efficient training.
'''
class RBM:
    
	def __init__(self, n_visible, n_hidden, lr=0.001, epochs=5, mode='bernoulli', batch_size=32, k=3, optimizer='adam', gpu=True, savefile=None, early_stopping_patience=5):
		self.mode = mode # bernoulli or gaussian RBM
		self.n_hidden = n_hidden #  Number of hidden nodes
		self.n_visible = n_visible # Number of visible nodes
		self.lr = lr # Learning rate for the CD algorithm
		self.epochs = epochs # Number of iterations to run the algorithm for
		self.batch_size = batch_size
		self.k = k
		self.optimizer = optimizer
		self.beta_1 = 0.9
		self.beta_2 = 0.999
		self.epsilon = 1e-7

		'''What is this m, v ??? '''
		self.m = [0, 0, 0]
		self.v = [0, 0, 0]
		self.m_batches = {0:[], 1:[], 2:[]}
		self.v_batches = {0:[], 1:[], 2:[]}

		self.savefile = savefile
		self.early_stopping_patience = early_stopping_patience
		self.stagnation = 0
		self.previous_loss_before_stagnation = 0
		self.progress = []



		if torch.cuda.is_available() and gpu==True:  
			dev = "cuda:0" 
		else:  
			dev = "cpu"  
		self.device = torch.device(dev)

		# Xavier initialization: std =  4*(6 / root(input  + ouput)) 
		std = 4 * np.sqrt(6.0 / (self.n_visible + self.n_hidden))
		self.W = torch.normal(mean = 0, std = std, size = (self.n_hidden , self.n_visible))  # (n_hidden x n_visible) weight matrix
		self.vb = torch.zeros(size = (1, n_visible), dtype = torch.float32) #visible layer bias -- (1 x n_visible)
		self.hb = torch.zeros(size = (1, n_hidden), dtype = torch.float32) #hidden layer bias -- (1 x n_hidden)


		# Move parameters to GPU for faster computation
		self.W = self.W.to(self.device)
		self.vb = self.vb.to(self.device)
		self.hb = self.hb.to(self.device)
	def sample_h(self, x):
		'''
		This function samples the hidden layer units given the visible layer input for the Restricted Boltzmann Machine (RBM).
		
		Explanation:
		
		- x: This represents the input to the RBM's hidden layer, which corresponds to the visible layer units from the previous step. In the context of the RBM, these are the units from the visible layer that are being used to compute the hidden layer activations.

		The formula used here to compute the probability of the hidden unit \(h_j\) being activated given the visible layer \(v\) is:
		
		p(h_j = 1 | v) = sigmoid(v * W^T + b_h)
		
		Where:
		- v is the visible layer's input, represented by the variable `x`.
		- W^T is the transpose of the weight matrix that connects the visible layer to the hidden layer. This is represented by `self.W.t()`, where `.t()` stands for transpose.
		- b_h is the bias of the hidden layer, represented by `self.hb`.
		- Sigmoid is the activation function that squashes the computed result to a value between 0 and 1, which gives the probability that a hidden unit will be activated (set to 1).

		The function first computes the weighted sum of the inputs from the visible layer and adds the hidden layer bias to it (this is represented by the variable `activation`).

		After calculating the activation, the sigmoid function is applied to compute the probability that the hidden units will be activated.

		Sampling Techniques:
		- Bernoulli Sampling: If the mode is set to 'bernoulli', this function uses Bernoulli sampling. It randomly samples the hidden units as either 1 or 0 based on the computed probabilities (p_h_given_v).
		- Gaussian Sampling: If the mode is not 'bernoulli', the function uses Gaussian sampling. It adds Gaussian noise (mean = 0, std = 1) to the computed probabilities (p_h_given_v) to create continuous values for the hidden units.
		
		Detailed steps:
		- wx = torch.mm(x, self.W.t()): This step performs the matrix multiplication between the input (`x`, the visible layer) and the transpose of the weight matrix (`self.W.t()`).
		- activation = wx + self.hb: Adds the bias of the hidden layer (`self.hb`) to the weighted input.
		- p_h_given_v = torch.sigmoid(activation): Applies the sigmoid activation function to compute the probability that each hidden unit will be activated.
		
		If the mode is 'bernoulli', it returns both the probability and the sampled values (0 or 1). If the mode is not 'bernoulli', it returns the probability and the Gaussian-sampled values.
		'''

		# x -- previous visible layer units given as inputs to the hidden layer
		# x -- input to the RBM's hidden layer --- Visible layer units, written as Vj in note
		wx = torch.mm(x, self.W.t)  # input * Weight_matrix.Transpose() 
		'''
		Here, the dimension of multiplication is 
			= (1 x n_visible) * (n_hidden x n_visible).Transpose
			= (1 x n_visible) * (n_visible x n_hidden)
		'''
		activation = wx + self.hb  # addition of the hidden layer bias
		p_h_given_v = torch.sigmoid(activation)  # POSITIVE EDGE COMPUTATION

		if self.mode == 'bernoulli':
			# Bernoulli's Sampling
			return p_h_given_v, torch.bernoulli(p_h_given_v)
		else:
			# Gaussian Sampling
			return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape))

	def sample_v(self, y):
		# Same as above except for visible layer
		# y -- output of previous RBM's hidden units or Init vector in case of 1st RBM

		wy = torch.mm(y, self.W)

		'''
		Here, the dimension of multiplication is 
			= (1 x n_hidden) * (n_hidden x n_visible)
		'''
		activation = wy + self.vb
		p_v_given_h = torch.sigmoid(activation)
		if self.mode == 'bernoulli':
			return p_v_given_h, torch.bernoulli(p_v_given_h)
		else:
			return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean = 0, std = 1,  size = p_v_given_h.shape))
	
	def adam(self, g, epoch, index):
		pass

