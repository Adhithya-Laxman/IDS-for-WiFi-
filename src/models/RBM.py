
import numpy as np
import torch
import random
from tqdm import trange

class RBM:
    
	def __init__(self, n_visible, n_hidden, lr=0.001, epochs=5, mode='bernoulli', batch_size=32, k=3, optimizer='adam', gpu=True, savefile=None, early_stopping_patience=5):
		self.mode = mode # bernoulli or gaussian RBM
		self.n_hidden = n_hidden #  Number of hidden nodes
		self.n_visible = n_visible # Number of visible nodes
		self.lr = lr # Learning rate for the CD algorithm
		self.epochs = epochs # Number of iterations to run the algorithm for
		self.batch_size = batch_size
		self.k = k #Number of steps in gibbs sampling
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
			print(f"Using device : {dev}")
		else:  
			dev = "cpu"  
			print(f"Using device : {dev}")

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

		The formula used here to compute the probability of the hidden unit (h_j) being activated given the visible layer (v) is:
		
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
		wx = torch.mm(x, self.W.t())  # input * Weight_matrix.Transpose() 
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
		'''
		Implements the Adam optimizer for updating model parameters.

		Args:
		- g (torch.Tensor): The gradient of the loss function with respect to the model parameter at the current step.
		- epoch (int): The current epoch or time step in the optimization process.
		- index (int): The index of the parameter being updated (0 for weights, 1 for visible biases, 2 for hidden biases).

		The Adam optimizer computes adaptive learning rates for each parameter by maintaining running averages of both the gradients (first moment estimate) and the squared gradients (second moment estimate). This helps mitigate issues like vanishing or exploding gradients and ensures stable training.

		The following steps are performed for each parameter being updated:

		1. **First moment estimate (m_t)**: This is the exponentially decaying average of past gradients.
		Formula: 
			m_t = beta_1 * m_{t-1} + (1 - beta_1) * g
		Where:
		- beta_1: The decay rate for the first moment estimate (typically 0.9).
		- m_{t-1}: The previous first moment estimate.
		- g: The current gradient.

		2. **Second moment estimate (v_t)**: This is the exponentially decaying average of past squared gradients.
		Formula:
			v_t = beta_2 * v_{t-1} + (1 - beta_2) * g^2
		Where:
		- beta_2: The decay rate for the second moment estimate (typically 0.999).
		- v_{t-1}: The previous second moment estimate.
		- g^2: The square of the current gradient.

		3. **Bias-corrected estimates**:
		Since both the first and second moment estimates (m_t and v_t) are initialized as zeros, they are biased towards zero, especially during the initial steps. To correct this bias, the following unbiased estimates are computed:

		- **Bias-corrected first moment estimate (m_hat)**:
			m_hat = m_t / (1 - beta_1^t)
		- **Bias-corrected second moment estimate (v_hat)**:
			v_hat = v_t / (1 - beta_2^t)

		Returns:
		- torch.Tensor: The update value for the parameter being optimized.

		Example:
		For updating the weight matrix W (index 0), visible bias vb (index 1), and hidden bias hb (index 2), this function is called separately for each during the training process.
		'''
		
		# First moment estimate update
		self.m[index] = self.beta_1 * self.m[index] + (1 - self.beta_1) * g

		# Second moment estimate update
		self.v[index] = self.beta_2 * self.v[index] + (1 - self.beta_2) * torch.pow(g, 2)

		# Bias-corrected first moment estimate
		m_hat = self.m[index] / (1 - np.power(self.beta_1, epoch)) + (1 - self.beta_1) * g / (1 - np.power(self.beta_1, epoch))

		# Bias-corrected second moment estimate
		v_hat = self.v[index] / (1 - np.power(self.beta_2, epoch))

		# Parameter update value
		'''
		The final update for the parameter is computed using the bias-corrected estimates. The learning rate is adjusted based on the ratio of the first moment estimate to the square root of the second moment estimate:
		
		Formula:
			update = m_hat / (sqrt(v_hat) + epsilon)
		Where:
		- epsilon: A small value to prevent division by zero (typically 1e-8).

		'''
		return m_hat / (torch.sqrt(v_hat) + self.epsilon)


	def update(self, v0, vk, ph0, phk, epoch):
		"""
		Updates the parameters (weights, visible biases, hidden biases) of the Restricted Boltzmann Machine (RBM) 
		based on the Contrastive Divergence (CD) algorithm. It calculates the gradients for the weights and biases 
		using the difference between the initial and reconstructed visible and hidden units and applies the Adam 
		optimizer if specified.

		Parameters:
		-----------
		v0 : torch.Tensor
			The initial visible layer units (input data) at the start of the contrastive divergence.
		vk : torch.Tensor
			The reconstructed visible layer units after k steps of Gibbs sampling.
		ph0 : torch.Tensor
			The probabilities of the hidden units being activated given the initial visible units v0. 
		phk : torch.Tensor
			The probabilities of the hidden units being activated given the reconstructed visible units vk.
		epoch : int
			The current training epoch, used to adjust the Adam optimizer's moment estimates.

		Returns:
		--------
		None
			The function directly updates the weights (`self.W`), visible biases (`self.vb`), and hidden biases (`self.hb`) 
			in place. No values are returned.

		Key Formulas:
		-------------
		- Weight gradient (dW):
			dW = (v0^T * ph0 - vk^T * phk)^T
			This formula computes the difference between the correlations of the visible and hidden units before and 
			after reconstruction, updating the weights to minimize reconstruction error.

		- Visible bias gradient (dvb):
			dvb = sum(v0 - vk)
			This computes the difference between the initial and reconstructed visible units and updates the visible biases 
			to reduce error in the visible layer.

		- Hidden bias gradient (dhb):
			dhb = sum(ph0 - phk)
			This computes the difference between the probabilities of hidden unit activations before and after reconstruction, 
			adjusting the hidden biases to improve the hidden layer's performance.
		"""
		dW = (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
		dvb = torch.sum((v0-vk), 0)
		dhb = torch.sum((ph0 - phk), 0)

		if self.optimizer == 'adam':
			dW = self.adam(dW, epoch, 0)
			dvb = self.adam(dvb, epoch, 1)
			dhb = self.adam(dhb, epoch, 2)
		
		# Update existing parameters

		self.W += self.lr * dW
		self.vb += self.lr * dvb
		self.hb += self.lr * dhb


	def train(self, dataset):
		"""
		Trains the RBM (Restricted Boltzmann Machine) model using the provided dataset.

		Parameters:
		-----------
		dataset : torch.Tensor
			The input data (visible layer units) to train the RBM, expected in torch Tensor format.
			The dataset is moved to the specified device (e.g., GPU or CPU).

		Training Process:
		-----------------
		- A progress bar (using `trange`) is displayed, showing the epoch number and the loss value at each epoch.
		- For each epoch:
			1. Initializes training loss and batch counter.
			2. Iterates through the dataset in batches:
				a. Selects batches `v0` (initial visible layer) and `vk` (reconstructed visible layer after Gibbs sampling).
				b. Performs `k` steps of Gibbs sampling (alternating between hidden and visible layers).
				c. Updates the weight matrix and biases using the gradients computed from the contrastive divergence.
				d. Computes the loss as the average absolute difference between `v0` and `vk` and accumulates it.
			3. Appends the average loss of the current epoch to the progress tracking list.
			4. Updates the progress bar description with epoch details (epoch number and rounded loss value).

		Early Stopping:
		---------------
		- If the loss stops improving (defined by the `early_stopping_patience` parameter), the training loop terminates early.
		- The loss is compared against `previous_loss_before_stagnation`. If no improvement is detected for `early_stopping_patience` consecutive epochs, training stops.

		Model Saving:
		-------------
		- After completing all epochs (or stopping early), the model parameters (weights `W`, visible biases `vb`, and hidden biases `hb`) are saved to a file if `savefile` is provided.
		
		Attributes Updated During Training:
		-----------------------------------
		- `self.W`: Updated weight matrix connecting visible and hidden units.
		- `self.vb`: Updated visible layer biases.
		- `self.hb`: Updated hidden layer biases.
		- `self.progress`: List of training loss values over all epochs.
		- `self.stagnation`: Counter tracking stagnation for early stopping.
		- `self.previous_loss_before_stagnation`: Loss value from previous epoch to monitor improvements.
		
		Parameters:
		-----------
		- `self.optimizer`: If 'adam', applies Adam optimizer for parameter updates.
		- `self.batch_size`: Size of batches used for training.
		- `self.k`: Number of Gibbs sampling steps per batch.
		- `self.epochs`: Total number of epochs to train the model.
		- `self.device`: Specifies the device (CPU/GPU) to use during training.

		Returns:
		--------
		None
		"""
		
		# Move dataset to the specified device (GPU/CPU)
		dataset = dataset.to(self.device)
		
		# Generate a progress bar for epochs using trange
		learning = trange(self.epochs, desc = str('Staring...'))  # Initialize progress bar
		
		for epoch in learning:
			train_loss = 0  # Initialize the loss for the current epoch
			counter = 0     # Initialize a counter for batches
			
			# Iterate through the dataset in batches
			for batch_start_index in range(0, dataset.shape[0] - self.batch_size, self.batch_size):
				# Get a batch for the visible layer (both v0 and vk start as the same)
				vk = dataset[batch_start_index : batch_start_index + self.batch_size]
				v0 = dataset[batch_start_index : batch_start_index + self.batch_size]
				
				# Sample hidden probabilities and states from the initial visible layer
				ph0, _ = self.sample_h(v0)

				# Perform Gibbs sampling 'k' times (alternating between hidden and visible layers)
				for k in range(self.k):
					# Sample hidden and visible layers alternately
					_, hk = self.sample_h(vk)  # Sample hidden states from the visible layer
					_, vk = self.sample_v(hk)  # Sample visible states from the hidden layer

				# After 'k' steps, sample the final hidden probabilities from the reconstructed visible layer
				phk, _ = self.sample_h(vk)
				
				# Update weights and biases using contrastive divergence
				self.update(v0, vk, ph0, phk, epoch + 1)
				
				# Compute the reconstruction loss as the average absolute difference between v0 and vk
				train_loss += torch.mean(torch.abs(v0 - vk))
				counter += 1  # Increment the batch counter

			# Append the average loss for this epoch to the progress list
			self.progress.append(train_loss.item() / counter)
			
			# Update the progress bar with the current epoch and loss
			details = {'epoch': epoch + 1, 'loss': round(train_loss.item() / counter, 4)}
			learning.set_description(str(details))  # Display current progress
			learning.refresh()  # Update the display

			# Early stopping check: If the loss is no longer improving, terminate training
			if train_loss.item() / counter > self.previous_loss_before_stagnation and epoch > self.early_stopping_patience + 1:
				self.stagnation += 1  # Increment stagnation counter
				if self.stagnation == self.early_stopping_patience - 1:
					learning.close()  # Stop progress bar
					print("Not improving, stopping training loop.")
					break
				else:
					self.previous_loss_before_stagnation = train_loss.item() / counter  # Update loss for next comparison
					self.stagnation = 0  # Reset stagnation counter

		print("All epochs completed")
		learning.close()  # Close the progress bar

		# Save the model parameters if a save file is provided
		if self.savefile is not None:
			model = {'W': self.W, 'vb': self.vb, 'hb': self.hb}
			torch.save(model, self.savefile)  # Save weights and biases to a file


	def load_rbm(self, save_file):
		"""
		Loads a pre-trained RBM model from a saved file.

		Parameters:
		-----------
		save_file : str
			The path to the file containing the saved model parameters (weights and biases).

		Returns:
		--------
		None
		"""
		# Load the model parameters from the file
		loaded = torch.load(save_file)

		# Assign the loaded parameters to the RBM instance
		self.W = loaded['W']
		self.vb = loaded['vb']
		self.hb = loaded['hb']

		# Move the parameters to the specified device (GPU/CPU)
		self.W = self.W.to(self.device)
		self.vb = self.vb.to(self.device)
		self.hb = self.hb.to(self.device)




'''
So, it it working! 
'''