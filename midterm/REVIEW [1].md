# **📌 Study Plan for AI Midterm**

## **1. Intelligent Agents**
### **Definition**
An **intelligent agent** is a system that perceives its environment through sensors and takes actions via actuators to maximize its performance.

### **Types of Intelligent Agents**
1. **Simple Reflex Agents** – Act only based on current perception (e.g., thermostat).
2. **Model-Based Reflex Agents** – Maintain an internal state (e.g., self-driving cars).
3. **Goal-Based Agents** – Make decisions to achieve specific goals (e.g., pathfinding AI).
4. **Utility-Based Agents** – Choose the best action using a utility function (e.g., AI in stock trading).
5. **Learning Agents** – Improve their performance over time (e.g., Deep Q-Networks in Reinforcement Learning).

### **Example: Self-Driving Car**
- **Sensors**: Cameras, LiDAR, GPS.
- **Actuators**: Steering, acceleration, braking.
- **Decision-Making**: Uses AI to interpret surroundings and navigate.

---

## **2. Pretrained Neural Networks**
### **Definition**
Pretrained networks are models trained on large datasets and reused for specific tasks.

### **Examples**
- **Image Classification**: **AlexNet, ResNet**  
- **NLP**: **GPT, BERT**  
- **Generative AI**: **CycleGAN (image-to-image transformation)**  
- **Image Captioning**: **NeuralTalk2**  

### **Advantages**
✅ Saves computation time  
✅ Requires less labeled data  
✅ Provides better generalization  

### **Fine-Tuning**
- Instead of training from scratch, modify some layers of a pretrained model.
- Example: **Using ResNet for medical imaging classification** by fine-tuning only the last layers.

---

## **3. Machine Learning Mechanisms**
### **Supervised Learning**
- **Uses labeled data** (e.g., classification, regression).
- **Examples**: Predicting housing prices, spam email detection.

### **Unsupervised Learning**
- **Finds patterns in unlabeled data** (e.g., clustering).
- **Examples**: Customer segmentation, anomaly detection.

### **Reinforcement Learning**
- **Agent learns by interacting with the environment**.
- **Example**: Chess AI learns moves by trial and error.

---

## **4. Neural Network Fundamentals**
### **Artificial Neuron**
- The basic unit of a neural network.
- Performs:
  1. **Linear Transformation**:  
     $$z = w_1 x_1 + w_2 x_2 + b$$
  2. **Activation Function**: Applies non-linearity.

### **Activation Functions**

| Function  | Formula | Purpose |
|-----------|------------|------------|
| **Sigmoid**  | \( \sigma(x) = \frac{1}{1+e^{-x}} \) | Used in probability-based outputs. |
| **Tanh**  | \( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \) | Used in zero-centered activation. |
| **ReLU**  | \( \max(0, x) \) | Prevents vanishing gradient. |

**Example:**  
If \( x = -3 \), what is ReLU(x)?  
$$ReLU(-3) = \max(0, -3) = 0$$

---

## **5. Layers in Neural Networks**
### **Types of Layers**
- **Input Layer**: Loads data into the model.
- **Hidden Layers**: Extract features using weights.
- **Output Layer**: Produces final predictions.

### **Dense Layer**
- Every neuron in the layer is connected to every neuron in the previous layer.
- **Implementation in PyTorch:**
  ```python
  import torch.nn as nn
  layer = nn.Linear(in_features=5, out_features=3)  # 5 inputs, 3 outputs
  ```

---

## **6. Weight Matrix & Bias Vector**
- **Weights**: Control how much influence an input has.
- **Biases**: Help shift the activation.

**Parameter Calculation**
- **Number of Weights** = \( \text{inputs} \times \text{neurons} \)
- **Number of Biases** = \( \text{neurons} \)
- **Total Parameters** = **weights + biases**

### **Example**
A **Dense Layer** has **4 inputs** and **6 neurons**.
$$
\text{Weights} = 4 \times 6 = 24
$$
$$
\text{Biases} = 6
$$
$$
\text{Total Parameters} = 24 + 6 = 30
$$

---

