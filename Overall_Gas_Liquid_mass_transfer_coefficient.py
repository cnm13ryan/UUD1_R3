#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# # Determine the overall gas-liquid mass transfer coefficient 
# 
# Assume that the gas-side mass transfer resistance is negligible as liquid is the continuous phase in the three-phase fluidized bed reactor. 
# 
# The overall gas-liquid mass transfer coefficient will be equal to the volumetric liquid-side mass transfer coefficient, $k_l a$, where $k_l$ is the liquid-side mass transfer coefficient and $a$ is the gas-liquid interfacial area per unit volume of the bed. 
# 
# In this case, we use an ANN approach, which takes into account of 6 literature reported models to correlate the parameter needed for our reactor design [1].

# In[21]:


def sigmoid(x):
    """Sigmoid activation function."""
    # Clip x to avoid overflow in exp
    # 709 is chosen because np.exp(709) is close to the max float value
    x_clipped = np.clip(x, -709, 709) 
    return 1 / (1 + np.exp(-x_clipped))

def linear(x):
    """Linear activation function."""
    return x

class SimpleNN:
    def __init__(self, weights_hidden, biases_hidden, weights_output, bias_output):
        """Initialize the neural network with weights and biases."""
        self.weights_hidden = weights_hidden
        self.biases_hidden = biases_hidden
        self.weights_output = weights_output
        self.bias_output = bias_output

    def forward_pass(self, x):
        """Perform a forward pass."""
        # Hidden layer
        z_hidden = np.dot(x, self.weights_hidden) + self.biases_hidden
        a_hidden = sigmoid(z_hidden)
        
        
        # Output layer
        z_output = np.dot(a_hidden, self.weights_output) + self.bias_output
        output = linear(z_output)
        
        return output

# Mass transfer parameters from the image
weights_hidden = np.array([
    [2.173, -0.087, 0.369, -0.353, -2.287],
    [-1.481, 0.937, -0.162, -1.244, 2.948],
    [-2.315, 1.186, -0.837, 0.620, -0.083],
    [1.042, 1.215, -1.050, 0.558, -4.776],
    [-3.180, 1.968, -1.054, 1.470, -0.721]
])

biases_hidden = np.array([-2.785, 1.203, -0.217, -2.255, 0.248])
weights_output = np.array([[0.831], [1.314], [-1.527], [0.845], [0.856]])
bias_output = np.array([-0.4126])

# Initialize the neural network with the provided weights and biases
nn = SimpleNN(weights_hidden, biases_hidden, weights_output, bias_output)


# In[24]:


# Input
input_size = 5  

# x = (d_p [m], \rho_p [kg / m^3], U_l [m/s], U_g [m/s], \varepsilon_s)
# d_p is the particle diameter [m]
# \rho_p is the particle density [kg / m^3]
# U_l is the gas velocity [m/s]
# U_g is the liquid velocity [m/s]
# \varepsilon_s is the solid holdup [n.d] 

x = np.array([3.5e-3, 1.72e5, 5, 66, 0.2])

# Perform a forward pass
output = nn.forward_pass(x)
print(f"Network output (k_l a): {output} [s^{-1}]")


# # References
# 
# [1]: F. S. Mjalli and A. Al-Mfargi, “Artificial Neural Approach for Modeling the Heat and Mass Transfer Characteristics in Three-Phase Fluidized Beds,” Industrial & Engineering Chemistry Research, vol. 47, no. 13, pp. 4542–4552, May 2008, doi: https://doi.org/10.1021/ie0715714.
