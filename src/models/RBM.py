
import numpy as np
import torch
import random
from tqdm import trange
'''
m,v in this file: 
In the context of your `RBM` class implementation, the variables `m` and `v` are used as part of the Adam optimization algorithm, specifically for maintaining the first and second moment estimates of the gradients. Here’s a breakdown of what they represent:

1. **`m`**: This variable represents the first moment estimate, which is the exponential moving average of the gradients. It captures the mean of the gradients over time. The update formula for `m` is:
   \[
   m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
   \]
   where:
   - \(m_t\) is the current first moment estimate,
   - \(m_{t-1}\) is the previous first moment estimate,
   - \(g_t\) is the current gradient,
   - \(\beta_1\) is the decay rate for the first moment (typically set to 0.9).

2. **`v`**: This variable represents the second moment estimate, which is the exponential moving average of the squared gradients. It captures the variance of the gradients over time. The update formula for `v` is:
   \[
   v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
   \]
   where:
   - \(v_t\) is the current second moment estimate,
   - \(v_{t-1}\) is the previous second moment estimate,
   - \(g_t^2\) is the square of the current gradient,
   - \(\beta_2\) is the decay rate for the second moment (typically set to 0.999).

### Initialization in Your Code

In your `RBM` class, `m` and `v` are initialized as lists with three elements, corresponding to the three parameters that the model will optimize (weights `W`, visible biases `vb`, and hidden biases `hb`). This structure allows for separate moment estimates for each of these parameters:

```python
self.m = [0, 0, 0]  # First moment estimates for W, vb, hb
self.v = [0, 0, 0]  # Second moment estimates for W, vb, hb
```

### Usage in Adam Optimization

When you call the `adam` method during the `update` phase of training, it calculates the moment estimates and then computes the adaptive learning rate based on these estimates. Here’s the relevant part of your code:

```python
if self.optimizer == 'adam':
    dW = self.adam(dW, epoch, 0)
    dvb = self.adam(dvb, epoch, 1)
    dhb = self.adam(dhb, epoch, 2)
```

Here, `adam` is called for each gradient, and it updates `m` and `v` accordingly for each parameter during the training process. 

This mechanism allows the Adam optimizer to adaptively adjust the learning rates for each parameter based on their individual gradient histories, leading to more efficient training.

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
		self.W = torch.normal(mean = 0, std = std, size = (self.n_hidden , self.n_visible))
		self.vb = torch.zeros(size = (1, n_visible), dtype = torch.float32)
		self.hb = torch.zeros(size = (1, n_hidden), dtype = torch.float32)


		self.W = self.W.to(self.device)
		self.vb = self.vb.to(self.device)
		self.hb = self.hb.to(self.device)
