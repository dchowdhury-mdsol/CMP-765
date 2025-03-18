# Midterm Review Topics

## What has been covered so far?

- [Intelligent Agent]()
- [Pretrained Neural Networks]()
- [Machine Learning Mechanism]()
- [Neural Network Fundamentals]()

## Terminologies

### Artifical Neuron
> A neural network consists of multiple layers of artificial neurons. Artificial neurons are the basic building blocks of Nueral Networks. An artificial neuron is constructed using two functions:
>   - Linear transformation
>   - Activation function

---

### Activation Functions
> An activation function is a non-linear function that represents a 'switch'. For some input, it produces a value close to 0 (the switch is off). For other inputs, it produces a significant value (the switch is on). The entire network would be reduced to a linear transformation if we remove activation functions from it's neurons.

Popular choices of activation functions:
- $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ - Chosen for the s-shaped graph.
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ - Chosen for the s-shaped graph.
- $\text{ReLU}(x) = \max(0, x)$ - Chosen for simplicity.

---

### Types of Layers
>   - Input Layer: Loads the data.
>   - Output Layer: Produces final predictions.
>   - Hidden Layer: Layers in the middle.
>   - Dense Layer: Full connected to the previous layer.

```python
# Defining a dense layer
import torch.nn as nn

DenseLayer = nn.Linear(in_features=10, out_features=5)

# Adding activation function to dense layer

## ReLU Activation
DenseReLU = nn.Sequential( # A compact way to stack layers
    nn.Linear(10, 5), # Fully connected layer
    nn.ReLU()         # Activation function
)

DenseTanH = nn.Sequential(
    nn.Linear(10, 5), # Fully connected layer
    nn.Tanh()         # Activation function
)
```

---

### Weight, Matrix, Bias Vector
Groups of parameters for a neural network. How to estimate the number of parameters:
- **Weights**: Control how much influence an input has.
- **Biases**: Help shift the activation.
- **Parameter Calculation**:
    - **Number of Weights** = $\text{inputs} \times \text{neurons}$
    - **Number of Biases** = $\text{neurons}$
    - **Total Parameters** = **weights + biases**
- **Example**:
    - A **Dense Layer** has **4 inputs** and **6 neurons**.

$\text{Weights} = 4 \times 6 = 24$

$\text{Biases} = 6$

$\text{Total Parameters} = 24 + 6 = 30$

---

### Loss Function
A function that measures the performance of the model.

$\text{MeanSquaredError} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$

> - Used for regression problems. 
> - Has an output range of (−∞,∞).
> - Excels with continous data types.

```python
# Example of MSE on a simple Linear regression
# Import required PyTorch libraries
import torch
import torch.nn as nn  # PyTorch's module for neural networks
import torch.optim as optim  # PyTorch's optimization library

# ---------------------------- Model Definition ----------------------------
# Define a simple Linear Regression model using nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize the parent class (nn.Module)
        
        # Define a single linear layer (fully connected)
        # It maps 1 input feature to 1 output feature
        self.linear = nn.Linear(in_features=1, out_features=1)  

    def forward(self, x):
        """
        Defines how data flows through the model.
        - Takes an input `x`, applies the linear transformation.
        - Returns the predicted output.
        """
        return self.linear(x)  # Linear function y = Wx + b

# ----------------------- DATA PREPARATION ------------------------

# Creating training data: a simple linear relationship (y = 2x + 3)
X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])  # Input values (independent variable)
Y_train = torch.tensor([[5.0], [7.0], [9.0], [11.0], [13.0]])  # Target values (y = 2x + 3)

# ----------------------- MODEL INITIALIZATION ------------------------

# Instantiate the linear regression model
model = LinearRegressionModel()

# Define Mean Squared Error (MSE) as the loss function
loss_fn = nn.MSELoss()

# Define the optimizer: Stochastic Gradient Descent (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)
# `model.parameters()` gives access to the weights (W) and bias (b) that the optimizer updates
# `lr=0.01` is the learning rate, which controls the step size for weight updates

# ----------------------- TRAINING LOOP ------------------------

num_epochs = 1000  # Number of training iterations (epochs)

for epoch in range(num_epochs):  # Loop through epochs

    # **Forward Pass: Compute model predictions**
    Y_pred = model(X_train)  # Model processes X_train and outputs predictions Y_pred
    
    # **Calculate Loss (How far is the prediction from the actual value?)**
    loss = loss_fn(Y_pred, Y_train)  # MSE = (1/N) * Σ (y_actual - y_predicted)²

    # **Zero out gradients from the previous step**
    optimizer.zero_grad()  # This prevents accumulation of gradients from previous iterations
    
    # **Backpropagation: Compute gradients of loss w.r.t. model parameters**
    loss.backward()  # Computes dL/dW and dL/db for updating model weights

    # **Gradient Descent Step: Update model parameters using the optimizer**
    optimizer.step()  # Adjusts weights W and bias b using computed gradients

    # **Print loss every 100 epochs for progress tracking**
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')  
        # `.item()` extracts the loss value from the tensor

# ----------------------- MODEL TESTING ------------------------

# Disable gradient tracking for evaluation (we don't need to compute gradients here)
with torch.no_grad():  
    test_input = torch.tensor([[6.0]])  # Test new input x = 6
    predicted_output = model(test_input)  # Model generates prediction

    # Print the predicted value for x=6
    print(f"Predicted Output for x=6: {predicted_output.item():.4f}")  
    # Expected output should be close to 15.0 (y = 2*6 + 3)
```

$\text{CrossEntropyLoss} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$

> - Used for classification problems. 
> - Has an output range of (0, 1). 
> - Excels with discrete class labels.

```python
# Example of CrossEntropy on a simple binary regression
# Import required PyTorch modules
import torch
import torch.nn as nn  # Neural network module
import torch.optim as optim  # Optimization module

# ---------------------------- Model Definition ----------------------------
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize the parent class (nn.Module)
        
        # Define a single linear layer (fully connected)
        # This model takes 1 input feature and produces 1 output (logit score)
        self.linear = nn.Linear(1, 1)  # 1 input → 1 output

    def forward(self, x):
        """
        Defines the forward pass of the model.
        - Takes input `x`, applies the linear transformation.
        - Returns raw logits (no activation function is applied).
        """
        return self.linear(x)  # Outputs logits (raw scores before activation)

# ---------------------------- Data Preparation ----------------------------

# Define input values (X_train)
X_train = torch.tensor([[0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5]])

# Define binary class labels (Y_train)
# - Class 0 if x < 2
# - Class 1 if x >= 2
Y_train = torch.tensor([[0], [0], [0], [1], [1], [1], [1]])  # Binary labels (0 or 1)

# ---------------------------- Model, Loss & Optimizer ----------------------------

# Instantiate the model
model = BinaryClassifier()

# Define Binary Cross-Entropy Loss with logits (logits → probabilities inside the function)
loss_fn = nn.BCEWithLogitsLoss()  

# Define the optimizer (Stochastic Gradient Descent) with a learning rate of 0.1
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ---------------------------- Training Loop ----------------------------

num_epochs = 1000  # Number of training iterations (epochs)

for epoch in range(num_epochs):  # Loop through each epoch
    
    # Forward Pass: Compute raw outputs (logits) before activation
    Y_logits = model(X_train)  

    # Compute Binary Cross-Entropy Loss
    loss = loss_fn(Y_logits, Y_train.float())  
    # Converts logits to probabilities using Sigmoid internally and applies BCE Loss:
    # L = - (y log(σ(ŷ)) + (1-y) log(1 - σ(ŷ)))

    # Zero out gradients from the previous step to prevent accumulation
    optimizer.zero_grad()
    
    # Compute gradients via backpropagation (derivatives of loss w.r.t. weights)
    loss.backward()
    
    # Update model parameters (weights and biases) using the computed gradients
    optimizer.step()

    #Print loss every 100 epochs to track progress
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')  # Extract loss value

# ---------------------------- Model Evaluation (Testing) ----------------------------

with torch.no_grad():  # Disable gradient computation during evaluation
    test_inputs = torch.tensor([[1.2], [2.2], [3.2]])  # Test new values
    
    # Get raw model outputs (logits)
    logits = model(test_inputs)  # Forward pass, output raw logits
    
    # Convert logits into probabilities using Sigmoid function
    predictions = torch.sigmoid(logits)  # Convert logits to probability scores
    
    # Apply threshold (>= 0.5) to classify predictions as Class 0 or 1
    predicted_classes = (predictions >= 0.5).float()  

    # Print test results (input, raw logit, probability, predicted class)
    for i, x in enumerate(test_inputs):
        print(f"Input: {x.item()}, Logit: {logits[i].item():.4f}, Probability: {predictions[i].item():.4f}, Predicted Class: {int(predicted_classes[i].item())}")
```
---

## Gradient Descent Algorithm
The gradient is the vector of the derivatives. The gradient helps us to efficiently reduce the loss. Ideally we want to find the parameter values that can achieve the minimum loss. The gradient descent algorithm repeatedly subtract the gradient from the parameter values. The model is updated as the parameter value changes.

**Steps**:
- Initialize parameters (weights $w$ and biases $b$) randomly or with small values.
- Compute the loss using a loss function (e.g., mean-squared-error, cross entropy).
- Compute the gradient (partial derivatives of the loss $w$.$r$.$t$. parameters).
- Update parameters using following:

   $
   w = w - \eta \frac{\partial L}{\partial w}
   $
   $
   b = b - \eta \frac{\partial L}{\partial b}
   $

   where **$\eta$** (learning rate) controls the step size.

- Repeat until convergence or until a stopping criterion is met.

---

### Example 1: Minimizing a Quadratic Function
Let's minimize $f(w) = (w - 3)^2$ using **gradient descent**.

1. Compute the gradient.

$\frac{d}{dw} (w - 3)^2 = 2(w - 3)$

2. Use update rule.

$w = w - \eta \cdot 2(w - 3)$

3. Implement with python.

```python
import numpy as np

# Initialize weight
w = np.random.randn()  # Random starting point
eta = 0.1  # Learning rate

# Perform Gradient Descent
for i in range(20):  # 20 iterations
    grad = 2 * (w - 3)  # Compute gradient
    w = w - eta * grad  # Update weight
    print(f"Iteration {i+1}: w = {w:.4f}")

print(f"Final optimized weight: {w:.4f}")
```
> Expected Result: $w$ approaches **3**, which is the minimum of $(w - 3)^2$.

---

### Example 2: Linear Regression using Gradient Descent
Consider a simple linear regression model:

$
y = w x + b
$

We aim to minimize **Mean Squared Error (MSE)**:

$
L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$

1. Compute the gradient:

$\frac{\partial L}{\partial w} = -\frac{2}{N} \sum x_i (y_i - \hat{y}_i)$

$\frac{\partial L}{\partial b} = -\frac{2}{N} \sum (y_i - \hat{y}_i)$

2. Update the weights.

$w = w - \eta \frac{\partial L}{\partial w}$

$b = b - \eta \frac{\partial L}{\partial b}$

3. Implement in python:

```python
import numpy as np

# Generate sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])  # y = 2x (true relationship)

# Initialize parameters
w = np.random.randn()
b = np.random.randn()
eta = 0.01  # Learning rate
epochs = 1000

# Perform Gradient Descent
for i in range(epochs):
    y_pred = w * X + b  # Predicted values
    loss = np.mean((Y - y_pred) ** 2)  # Compute MSE
    
    # Compute gradients
    grad_w = -2 * np.mean(X * (Y - y_pred))
    grad_b = -2 * np.mean(Y - y_pred)

    # Update weights
    w -= eta * grad_w
    b -= eta * grad_b

    if i % 100 == 0:  # Print every 100 iterations
        print(f"Epoch {i}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

print(f"Final optimized w = {w:.4f}, b = {b:.4f}")
```
> Expected Result: $w \approx 2, b \approx 0$, which correctly models $y = 2x$.

### Applications
- Image Classifications
    - AlexNet, ResNet
- Natural Language Processing
    - NeuralTalk2, GPT
- Generative AI
    - CycalGAN

## How to Build and Train a Neural Network

### Create a class that extends nn.Module

```python
import torch.nn as nn  # Import PyTorch neural network module

class NeuralNetwork(nn.Module):  # Define a neural network class by extending nn.Module
    def __init__(self):
        super().__init__()  # Call the parent constructor (nn.Module)

        # This layer flattens input data from a 2D tensor to a 1D vector.
        # Example: A 28x28 image (used in MNIST) will become a 784-dimensional vector.
        self.flatten = nn.Flatten()  

        # Define a stack of dense (fully connected) layers using nn.Sequential
        self.linearReLUStack = nn.Sequential(
            nn.Linear(28*28, 512),  # First dense layer (input: 784, output: 512)
            nn.ReLU(),              # Apply ReLU activation function
            nn.Linear(512, 512),    # Second dense layer (512 → 512 neurons)
            nn.ReLU(),              # Apply ReLU activation again
            nn.Linear(512, 10)      # Output layer (512 → 10 neurons for classification)
            # 10 nodes correspond to the 10 classes in MNIST (digits 0-9)
        )

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        - Takes input `x`, flattens it, then passes it through the dense layers.
        """
        x = self.flatten(x)           # Flatten input tensor (e.g., 28x28 → 784)
        return self.linearReLUStack(x)  # Pass data through the neural network


device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
model = NeuralNetwork().to(device)  # Create model and move it to the correct device (CPU/GPU)



# Create a model via nn.Sequential
model = nn.Sequential(
    nn.Flatten(),       # Flattens input (e.g., 28x28 → 784)
    nn.Linear(28*28, 512),  # Fully connected layer (784 → 512)
    nn.ReLU(),              # Activation function
    nn.Linear(512, 512),    # Fully connected layer (512 → 512)
    nn.ReLU(),              # Activation function
    nn.Linear(512, 10)      # Output layer (512 → 10)
)
```

> Key differences between `nn.Sequential` and a custom class `nn.Module`

Feature | `nn.Module` class | `nn.Sequential`
------- | ----------------- | ---------------
Flexibility | More control over forward pass. | Fixed layer structure.
Readability | Explicitly defines layers and forward pass. | More compact.
Custom Operations | Can include complex operations | Limited to stacking layers.

## Train a Neural Network using PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD  # Importing the Stochastic Gradient Descent optimizer

# Define the training loop function
def training_loop(n_epochs: int, optimizer: SGD, model: nn.Module, loss_fn, 
                  t_u_train, t_u_val, t_c_train, t_c_val):
    """
    Trains a PyTorch model using SGD optimizer.
    
    Parameters:
    - n_epochs (int): Number of epochs for training.
    - optimizer (SGD): The optimizer responsible for updating model weights.
    - model (nn.Module): The neural network model being trained.
    - loss_fn: Loss function (e.g., Mean Squared Error).
    - t_u_train: Training input data (temperature in unknown scale).
    - t_u_val: Validation input data.
    - t_c_train: Training output data (true temperature in Celsius).
    - t_c_val: Validation output data.
    """

    for epoch in range(1, n_epochs + 1):  # Loop through each epoch (starting from 1)
        
        # Forward Pass: Get model predictions for training data
        t_p_train = model(t_u_train)  # Model processes input t_u_train, outputs predictions
        
        # Compute Training Loss (MSE)
        loss_train = loss_fn(t_p_train, t_c_train)  # Calculate difference between predictions and true values
        
        # Zero out gradients from previous iteration (avoids accumulation)
        optimizer.zero_grad()

        # Back propagation: Compute gradients w.r.t. model parameters
        loss_train.backward()  # Computes dL/dW for every parameter W in the model
        
        # Gradient Descent Step: Update parameters
        optimizer.step()  # Applies weight updates using the computed gradients
        
        # Corrected: if epoch == 1 or epoch % 1000 == 0:
        if epoch == 1 or epoch % 1000 == 0:  # Print loss every 1000 epochs (or on the first epoch)
            
            # Forward Pass: Get model predictions for validation data
            t_p_val = model(t_u_val)  # Model makes predictions on validation set
            
            # Compute Validation Loss
            loss_val = loss_fn(t_p_val, t_c_val)  # Compare predictions with actual values
            
            # Print progress: Current epoch, training loss, and validation loss
            print(f'Epoch: {epoch}, Training Loss: {loss_train.item():.4f}, Validation Loss: {loss_val.item():.4f}')

# **Initializing the Optimizer (Stochastic Gradient Descent)**
optimizer = optim.SGD(seq_model.parameters(), lr=0.001)  
# `seq_model.parameters()` gives model weights to optimize
# `lr=0.001` sets learning rate (step size for weight updates)

# **Calling the Training Function**
training_loop(
    n_epochs=5000,  # Run for 5000 epochs
    optimizer=optimizer,  # Use the defined optimizer (SGD)
    model=seq_model,  # The model being trained (assumed already defined)
    loss_fn=nn.MSELoss(),  # Mean Squared Error (MSE) loss function
    t_u_train=t_un_train,  # Training input (uncalibrated temperature values)
    t_u_val=t_un_val,  # Validation input
    t_c_train=t_c_train,  # Training output (actual Celsius values)
    t_c_val=t_c_val  # Validation output
)
```