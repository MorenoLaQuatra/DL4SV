# Introduction to Deep Learning

```{warning}
This chapter is by no means a complete introduction to deep learning. It is just a brief overview to refresh the main concepts.
```

> Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction. 
{cite:ps}`lecun2015deep`

## Why Deep Learning for Speech and Vision?

Before diving into the technical details, it is important to understand why deep learning has become the dominant approach for speech and vision tasks.

**Traditional approaches** relied on hand-crafted features. For example, in computer vision, researchers would manually design filters to detect edges, corners, and textures. In speech processing, features like MFCCs (Mel-Frequency Cepstral Coefficients) were carefully engineered based on domain knowledge. While these approaches worked reasonably well, they had several limitations:

- **Manual feature engineering** requires extensive domain expertise and is time-consuming
- **Features may not be optimal** for the specific task at hand
- **Difficult to adapt** to new domains or tasks
- **Limited capacity** to capture complex patterns in data

Deep learning addresses these limitations by **automatically learning features** from raw or minimally processed data. Instead of designing features manually, we design the architecture and let the network learn the most useful representations through training.

**Why does this work?** Deep learning models with multiple layers can learn hierarchical representations:
- **Lower layers** learn simple patterns (edges, basic sounds)
- **Middle layers** combine simple patterns into more complex ones (shapes, phonemes)
- **Higher layers** learn abstract concepts (objects, words)

This hierarchical learning is particularly powerful for speech and vision because these domains naturally have this kind of structure. An image contains edges that form shapes that form objects. Speech contains phonemes that form syllables that form words.

## What is Deep Learning?

Deep Learning is a subfield of Machine Learning that involves the design, training, and evaluation of deep neural networks. While the term "deep" is not formally defined, it is generally used to refer to neural networks with more than one hidden layer.

### What is a Neural Network?

A neural network is a computational model inspired by the human brain. It is composed of a set of interconnected processing units, called neurons, that are organized in layers. Each neuron receives one or more inputs, performs a computation, and produces an output. The output of a neuron is then passed as input to other neurons. The output of the last layer is the output of the network.

The following figure shows a simple neural network with a single hidden layer. The network is composed of three layers:
- **input layer** is the first layer of the network. It receives the input data and passes it to the next layer.
- **hidden layer** is the middle layer of the network. It receives the input from the previous layer, performs a computation, and passes it to the next layer.
- **output layer** is the last layer of the network. It receives the input from the previous layer, performs a computation, and produces the output of the network.

```{figure} ./images/1_dl_intro/simple_nn.png
---
width: 60%
name: nn
alt: nn
---
```

This kind of neural network is called **feedforward neural network** because the information flows **from the input layer** to the **output** layer without any feedback connections.
Also, the network presented in the figure is called **fully-connected neural network** because each neuron in a layer is connected to all neurons in the next layer.

### What is a Deep Neural Network?

A deep neural network is a neural network with more than one hidden layer. The following figure shows a deep neural network with two hidden layers.

```{figure} ./images/1_dl_intro/deep_nn.png
---
width: 70%
name: deep_nn
alt: deep_nn
---
```

Similarly to the *shallow* case, the information flows from the input layer to the output layer without any feedback connections. Also, the network is fully-connected.

### Why "Deep" Matters: A Concrete Example

To understand why depth is important, consider a simple task: detecting a face in an image.

**Shallow network approach:**
- The network must learn to recognize faces directly from raw pixels
- It needs to learn all possible variations (different poses, lighting, expressions) in a single step
- This requires an enormous number of parameters and training examples

**Deep network approach:**
- **Layer 1**: Learns to detect edges (horizontal, vertical, diagonal)
- **Layer 2**: Combines edges into simple shapes (circles, rectangles)
- **Layer 3**: Combines shapes into facial features (eyes, nose, mouth)
- **Layer 4**: Combines features into face patterns

Each layer builds on the previous one, making the learning task easier at each step. This is the key insight behind deep learning: **complex functions can be more efficiently represented by composing simpler functions**.

```{admonition} Exercise: How many connections?
:class: tip, dropdown
What is the number of connections in a fully-connected neural network with $n$ input neurons, $m$ hidden neurons, and $k$ output neurons?

The number of total connections is given by:

$$
\sum_{i=1}^{l-1} n_i \times n_{i+1}
$$

where $l$ is the number of layers, $n_i$ is the number of neurons in layer $i$, and $n_{i+1}$ is the number of neurons in layer $i+1$.

What about the example in the figures above? How many connections are there in the shallow and deep neural networks?

- Shallow neural network ( 3 - 4 - 2 ): $3 \times 4 + 4 \times 2 = 20$
- Deep neural network ( 3 - 4 - 4 - 2 ): $3 \times 4 + 4 \times 4 + 4 \times 2 = 36$
```

# Supervised Learning

```{admonition} Learning Objectives
:class: learning-objectives
By the end of this section, you will understand:
- The difference between supervised, unsupervised, and reinforcement learning
- How to formulate a supervised learning problem
- The role of loss functions in training neural networks
- The optimization process using gradient descent and backpropagation
- How to split data into training, validation, and test sets
- Common evaluation metrics for classification and regression tasks
```

Deep learning models need to be *trained* on a collection of data to learn a specific task. The training process is based on the *learning paradigm* used to train the model. There are three main learning paradigms:
- **Supervised learning** the training data is composed of input-output pairs. The goal of the model is to learn a function that maps the input to the output.
- **Unsupervised learning** the training data is composed of input data only. The goal of the model is to learn the underlying structure of the data.
- **Reinforcement learning** the training data is composed of input data and a reward signal. The goal of the model is to learn a policy that maximizes the reward.

```{note}
In this course we will focus on supervised learning, which is the most common learning paradigm in deep learning.
While *unsupervised* and *reinforcement learning* are very important topics in deep learning, they are out of the scope of this course.
```

**Supervised Learning** (and self-supervised learning) is, at the moment, the most common learning paradigm in deep learning. The *supervision* is given by the training data, which is composed of input-output pairs. To formalize the problem, we can define the training data as a set of $N$ input-output pairs:

$$
\mathcal{D} = \{(\mathbf{x}_1, \mathbf{y}_1), (\mathbf{x}_2, \mathbf{y}_2), \dots, (\mathbf{x}_N, \mathbf{y}_N)\}
$$

where $\mathbf{x}_i$ is the input and $\mathbf{y}_i$ is the output for the $i$-th pair.

The goal of the model is to learn a function $f$ that maps the input $\mathbf{x}$ to the output $\mathbf{y}$:

$$
\mathbf{y} = f(\mathbf{x})
$$

In the context of deep learning, **the function $f$ is implemented as a neural network**. 

## Loss Function

The training process consists in finding the parameters of the neural network that minimize a loss function $\mathcal{L}$. It measures the difference between the predicted output $\mathbf{\hat{y}}$ and the ground truth output $\mathbf{y}$:

$$
\mathcal{L}(\mathbf{y}, \mathbf{\hat{y}}) = \mathcal{L}(f(\mathbf{x}), \mathbf{y})
$$

The loss function is a measure of how good the model is at predicting the output $\mathbf{y}$ given the input $\mathbf{x}$. The goal of the training process is to find the parameters of the neural network that minimize the loss function.

Some examples of loss functions are:
- **mean squared error** $\mathcal{L}(\mathbf{y}, \mathbf{\hat{y}}) = \frac{1}{N} \sum_{i=1}^N (\mathbf{y}_i - \mathbf{\hat{y}}_i)^2$. It is used for regression problems (e.g., predict a single real valued number - house price).
- **cross-entropy** $\mathcal{L}(\mathbf{y}, \mathbf{\hat{y}}) = - \sum_{i=1}^N \mathbf{y}_i \log(\mathbf{\hat{y}}_i)$. It is used for classification problems (e.g., predict a probability distribution over a set of classes - image classification).


## Optimization

A neural network is composed of neurons and connections between neurons. Each connection has a weight $w_i$ that controls the information flow between neurons.

```{figure} ./images/1_dl_intro/weights.png
---
width: 70%
name: weights
alt: weights
```

In the figure $w_1, w_2, w_3$ are the weights of the connections between the preceding neurons and the current neuron. The weights are the parameters of the neural network. The goal of the training process is to find the optimal values for the weights that minimize the loss function.

Defined in this way, the neural network is a function $f$ that maps the input $\mathbf{x}$ to the output $\mathbf{y}$ using a *linear combination* of the input and the weights:

$$
\mathbf{y} = f(\mathbf{x}, \mathbf{w}) = \sum_{i=1}^N w_i \mathbf{x}_i
$$

However, in practice, we are interested in learning more complex functions. To this end, we need to introduce a **non-linear activation function** $g$ that is applied to the output of each neuron: 

$$
\begin{align}
\mathbf{y} &= g(\mathbf{x}, \mathbf{w}) \\
&= g(\sum_{i=1}^N w_i \mathbf{x}_i)
\end{align}
$$

The addition of a non-linear activation function introduces non-linearity to the model. This is **essential** because real-world data is often **complex and non-linear**. Without non-linear activation functions, the neural network would be limited to representing linear mappings, which are not sufficient for capturing intricate relationships in data.

Some examples of activation functions are:
- **sigmoid** $g(x) = \frac{1}{1 + e^{-x}}$
- **tanh** $g(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- **ReLU** $g(x) = \max(0, x)$
- **Leaky ReLU** $g(x) = \max(0.01x, x)$

````{admonition} Code: Activation Functions
:class: tip, dropdown

The following code snippet draws the activation functions described above.

```python
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

x = np.linspace(-5, 5, 100)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot sigmoid in the first subplot
axs[0, 0].plot(x, y_sigmoid, label='sigmoid')
axs[0, 0].set_title('Sigmoid')
axs[0, 0].axhline(0, color='gray', linewidth=0.5)
axs[0, 0].axvline(0, color='gray', linewidth=0.5)

# Plot tanh in the second subplot
axs[0, 1].plot(x, y_tanh, label='tanh')
axs[0, 1].set_title('Tanh')
axs[0, 1].axhline(0, color='gray', linewidth=0.5)
axs[0, 1].axvline(0, color='gray', linewidth=0.5)

# Plot ReLU in the third subplot
axs[1, 0].plot(x, y_relu, label='ReLU')
axs[1, 0].set_title('ReLU')
axs[1, 0].axhline(0, color='gray', linewidth=0.5)
axs[1, 0].axvline(0, color='gray', linewidth=0.5)

# Plot Leaky ReLU in the fourth subplot
axs[1, 1].plot(x, y_leaky_relu, label='Leaky ReLU')
axs[1, 1].set_title('Leaky ReLU')
axs[1, 1].axhline(0, color='gray', linewidth=0.5)
axs[1, 1].axvline(0, color='gray', linewidth=0.5)

# Add legend to each subplot
for ax in axs.flat:
    ax.legend()

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('activation_functions_subplots_with_axes.png')

# Show the plot
plt.show()
```

```{figure} ./images/1_dl_intro/activations.png
---
width: 100%
name: activations
alt: activations
---
```
````


**How to find the optimal values for the weights?**

The training process is based on the *gradient descent* algorithm (or some variant of it). 
The gradient descent algorithm is an iterative algorithm that updates the weights of the neural network at each iteration. The update rule is the following:

$$
w_i = w_i - \alpha \frac{\partial \mathcal{L}}{\partial w_i}
$$

where $\alpha$ is the learning rate, which controls the size of the update. The learning rate is a hyperparameter of the model. The partial derivative $\frac{\partial \mathcal{L}}{\partial w_i}$ is the gradient of the loss function with respect to the weight $w_i$.

**Backpropagation** is an algorithm that computes the gradient of the loss function with respect to the weights of the neural network. It is based on the **chain rule of calculus**. The backpropagation algorithm is used to compute the gradient of the loss function with respect to the weights of the neural network.

The optimization process to train a neural network is the following:
1. Initialize the weights of the neural network
2. **Forward propagation**: compute the output of the neural network given the input $\mathbf{x}$ and the weights $\mathbf{w}$
3. Compute the loss function $\mathcal{L}(\mathbf{y}, \mathbf{\hat{y}})$
4. **Backward propagation**: compute the gradient of the loss function with respect to the weights of the neural network
5. Update the weights of the neural network using the gradient descent algorithm
6. Repeat steps 2-5 until the loss function is minimized

The following figure shows the training process of a neural network. The training data is composed of input-output pairs. The neural network is initialized with random weights. The training process consists in updating the weights of the neural network to minimize the loss function.

```{figure} ./images/1_dl_intro/training_process.png
---
width: 100%
name: training_process
alt: training_process
---
Sketch of the training process of a neural network. Image from [medium](https://medium.com/data-science-365/overview-of-a-neural-networks-learning-process-61690a502fa)
```

### Activation Functions and Gradients

The choice of the activation function is very important because it affects the training process. In particular, the activation function must be **differentiable** because we need to compute the gradient of the loss function with respect to the weights of the neural network.

Gradients of the loss function with respect to the weights are crucial for the backpropagation algorithm, which updates the weights in the direction that minimizes the loss. Therefore, the activation function must have the following properties:
- **Differentiability**: the activation function must be differentiable for all values of its input.
- **Continuous gradients**: the gradient of the activation function must be continuous for all values of its input. Continuous gradients facilitate stable weight updates during training.
- **Non-zero gradients**: the gradient of the activation function must be non-zero for all values of its input. 
- **Smoothness**: the smoothness of the activation function allows consistent weight updates during training.

One problem during the training process is the **vanishing gradient**. The vanishing gradient problem occurs when the gradient of the activation function is close to zero. In this case, the weights are not updated during the training process. Similarly, the **exploding gradient** problem occurs when the gradient of the activation function is very large. In this case, the weights are updated too much during the training process.
The choice of the activation function could mitigate those problems.

### Weight Initialization

Before training a neural network, we need to initialize the weights. Random initialization is necessary to break symmetry (if all weights are the same, all neurons would learn the same function). However, the way we initialize weights significantly affects training.

**Common initialization strategies:**

- **Xavier (Glorot) initialization**: Designed for sigmoid and tanh activations. Weights are initialized from a distribution with variance $\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$ where $n_{in}$ is the number of input units and $n_{out}$ is the number of output units.

- **He initialization**: Designed for ReLU activations. Weights are initialized from a distribution with variance $\text{Var}(w) = \frac{2}{n_{in}}$.

**Why does initialization matter?** Poor initialization can lead to:
- Vanishing gradients (weights too small)
- Exploding gradients (weights too large)
- Slow convergence
- Getting stuck in poor local minima

Modern deep learning frameworks (like PyTorch) use appropriate initialization by default, but understanding this concept is important when designing custom layers or debugging training issues.

### Optimization of a complex model

In deep learning:
- the neural network is composed of multiple layers
- each layer is composed of multiple neurons
- each neuron has multiple weights

The number of parameters in real-world neural networks grows very quickly. For this reason, while optimizing the parameters of a neural network, we usually parallelize the computation of the gradient. 
In other words, instead of having a single training example $\mathbf{x_i}$, we use a batch of training examples $\mathbf{X} = \{\mathbf{x_1}, \mathbf{x_2}, \dots, \mathbf{x_N}\}$ to compute the gradient. 
This approach is called **mini-batch gradient descent**. The size of the mini-batch $N$ is a hyperparameter of the model.

```{figure} ./images/1_dl_intro/training_batch.png
---
width: 80%
name: training_batch
alt: training_batch
---
Training process with mini-batch. A complete iteration over the training data is called **epoch**.
```

## Overfitting and Regularization

One of the most important challenges in deep learning is **overfitting**. Overfitting occurs when a model learns the training data too well, including its noise and peculiarities, and fails to generalize to unseen data.

**Signs of overfitting:**
- Training loss continues to decrease while validation loss increases
- High accuracy on training data but poor accuracy on validation/test data
- Model memorizes training examples instead of learning general patterns

### Regularization Techniques

Regularization techniques help prevent overfitting by constraining the model during training:

**1. L1 and L2 Regularization**

Add a penalty term to the loss function based on the magnitude of weights:

- **L2 regularization (weight decay)**: $\mathcal{L}_{total} = \mathcal{L} + \lambda \sum_i w_i^2$
  - Encourages small weights
  - Most commonly used
  - Implemented as weight decay in optimizers

- **L1 regularization**: $\mathcal{L}_{total} = \mathcal{L} + \lambda \sum_i |w_i|$
  - Encourages sparse weights (many weights become exactly zero)
  - Useful for feature selection

**2. Dropout**

During training, randomly "drop" (set to zero) a fraction of neurons with probability $p$ (typically 0.2 to 0.5). This forces the network to learn redundant representations and prevents co-adaptation of neurons.

```python
import torch.nn as nn

# During training
x = nn.Dropout(p=0.5)(x)  # 50% of neurons are randomly set to zero

# During evaluation
model.eval()  # Dropout is automatically disabled
```

**3. Early Stopping**

Stop training when validation performance stops improving, even if training loss is still decreasing. This prevents the model from overfitting to the training data.

**4. Data Augmentation**

Artificially increase the size of the training dataset by applying transformations (for images: rotation, flipping, cropping; for audio: time stretching, pitch shifting). This helps the model learn invariances and reduces overfitting.

### Batch Normalization

**Batch Normalization** is a technique that normalizes the inputs of each layer, making training faster and more stable. For a mini-batch, it normalizes each feature to have mean 0 and variance 1:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

where $\mu_B$ is the batch mean, $\sigma_B^2$ is the batch variance, and $\epsilon$ is a small constant for numerical stability.

**Benefits of Batch Normalization:**
- **Faster training**: allows higher learning rates
- **Reduces sensitivity** to initialization
- **Acts as regularization**: reduces the need for dropout
- **Stabilizes training**: reduces internal covariate shift

**Where to apply:** Typically applied after linear/convolutional layers and before activation functions.

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Normalize before activation
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

```{admonition} Common Pitfall
:class: warning
Batch normalization behaves differently during training and evaluation. During training, it uses batch statistics. During evaluation, it uses running averages computed during training. Always remember to switch between `model.train()` and `model.eval()` modes.
```

## Training and Validation

The training process involves the definition of the following components:
- neural network architecture
- loss function
- optimization algorithm
- hyperparameters (e.g., learning rate, batch size, etc.)
- **dataset(s)**
- metrics to evaluate the model

In the *superised learning* setting, the training data is composed of input-output pairs. We are interested in training a model that is able to **generalize** to unseen data. To this end, we need to split the training data into (at least) two sets:
- **training set** is used to train the model
- **test set** is used to evaluate the model on unseen data

In principle, those two splits should be enough to train and evaluate the model. However, in practice, we need to **tune the hyperparameters** of the model. In other words, we need to find the best set of hyperparameters that minimize the loss function on the test set. To this end, we need to split the training set into two sets:
- **Training set** is used to train the model.
- **validation set** is used to evaluate the model during the training process (e.g., at each epoch) and it is used to tune the  set of hyperparameters (e.g., the learning rate $\alpha$, the number of epochs, etc.).

```{figure} ./images/1_dl_intro/train_val_test.png
---
width: 80%
name: train_val_test
alt: train_val_test
---
Split of the training data into training, validation, and test sets.
Image from [towards data science](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)
```

Considering the data split, the training process now involves the following steps:
1. Split the training data into training, validation, and test sets.
2. Train the model on the training set.
3. Evaluate the model on the validation set.
4. Tune the hyperparameters of the model.
5. Repeat steps 2-4 until the final model is selected.
6. Evaluate the final model on the test set.

```{note}
In *traditional machine learning* is common to used cross-validation to tune the hyperparameters of the model. However, in deep learning, the training process is computationally expensive and the data is usually large. For this reason, we usually split the data into **fixed** training, validation, and test sets.
```

````{admonition} Code: Train-Validation-Test Split
:class: tip, dropdown

The following code snippet shows how to split the data into training, validation, and test sets using [scikit-learn](https://scikit-learn.org/stable/).

```python
from sklearn.model_selection import train_test_split

X, y = load_data() # Load the data

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
```
````

## Evaluation Metrics

The evaluation of a model is based on the **evaluation metric(s)**. The evaluation metric is a function that measures the performance of the model on the test set. The choice of the evaluation metric depends on the task we are trying to solve. 

**Classification** is the task of predicting a class label given an input. Some commonly used evaluation metrics for classification are:
- **accuracy** is the ratio between the number of correct predictions and the total number of predictions. It is the most common evaluation metric for classification problems. 
- 
$$
\text{accuracy} = \frac{\text{number of correct predictions}}{\text{total number of predictions}}
$$

- **precision** is the ratio between the number of true positives and the number of true positives and false positives. It measures the ability of the model to correctly predict the positive class.

$$
\text{precision} = \frac{\text{true positives}}{\text{true positives} + \text{false positives}}
$$

- **recall** is the ratio between the number of true positives and the number of true positives and false negatives. It measures the ability of the model to correctly predict the positive class.

$$
\text{recall} = \frac{\text{true positives}}{\text{true positives} + \text{false negatives}}
$$

- **F1-score** is the harmonic mean of precision and recall. It is a single metric that combines precision and recall.

$$
\text{F1-score} = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}
$$

Many other evaluation metrics are available for classification problems. For example, the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix), the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic), and the [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) are commonly used to evaluate classification models.

**Regression** is the task of predicting a real-valued number given an input. Some commonly used evaluation metrics for regression are:
- **mean squared error (MSE)** is the average of the squared differences between the predicted value and the ground truth value. It is the most common evaluation metric for regression problems.

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N (\mathbf{y}_i - \mathbf{\hat{y}}_i)^2
$$

- **mean absolute error** is the average of the absolute differences between the predicted value and the ground truth value.

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\mathbf{y}_i - \mathbf{\hat{y}}_i|
$$

- **root mean squared error (RMSE)** is the square root of the average of the squared differences between the predicted value and the ground truth value.

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\mathbf{y}_i - \mathbf{\hat{y}}_i)^2}
$$

````{admonition} Code: Evaluation Metrics
:class: tip, dropdown

In python, there are different packages that implement the evaluation metrics described above. For example, [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html) supports many evaluation metrics for classification and regression problems.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

# Compute accuracy
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
```

````

Many other evaluation metrics are available for regression problems. For example, the [mean absolute percentage error (MAPE)](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) is commonly used to evaluate regression models.

## Computational Considerations

When working with deep learning models, computational resources are a critical consideration. Understanding these concepts helps you design efficient experiments and manage resources effectively.

### GPU vs CPU Training

**Why use GPUs?** Deep learning involves many matrix operations that can be parallelized. GPUs have thousands of cores optimized for parallel computation, making them much faster than CPUs for training neural networks.

**Speed comparison:** A typical GPU can be 10-100x faster than a CPU for training deep neural networks, depending on the model size and batch size.

**Memory considerations:**
- **Model parameters**: The weights of the neural network (e.g., a model with 100M parameters using 32-bit floats requires ~400MB)
- **Activations**: Intermediate values computed during forward pass (depends on batch size and model architecture)
- **Gradients**: Same size as parameters, stored during backpropagation
- **Optimizer state**: Optimizers like Adam store additional information (momentum, variance) for each parameter

**Rule of thumb:** Total GPU memory needed is roughly: $\text{memory} \approx 4 \times \text{model size} \times \text{batch size}$

### Batch Size Considerations

**Larger batch sizes:**
- More memory required
- Faster training (better GPU utilization)
- May lead to worse generalization (sharp minima)
- More stable gradients

**Smaller batch sizes:**
- Less memory required
- Slower training
- May lead to better generalization (flat minima)
- Noisier gradients (can help escape local minima)

**Common practice:** Start with the largest batch size that fits in memory, then reduce if needed.

### Training Time Estimation

To estimate training time:

1. Measure time per batch on your hardware
2. Calculate: $\text{time} = \text{time per batch} \times \text{batches per epoch} \times \text{epochs}$
3. Add overhead for validation and checkpointing

**Example:** For a dataset with 50,000 samples, batch size 32, and 100 epochs:
- Batches per epoch: 50,000 / 32 = 1,563
- If each batch takes 0.1 seconds: 1,563 × 0.1 × 100 = 4.3 hours

```{admonition} Practical Tip
:class: tip
Always run a few training iterations first to estimate the time per batch before committing to a long training run. This helps you plan your experiments and avoid surprises.
```

# Conclusion

In this chapter, we refreshed the main concepts of deep learning. In particular, we introduced the main components of a deep learning model and the training process. We also introduced the main evaluation metrics for classification and regression problems.

**Key Takeaways:**

1. **Deep learning** learns hierarchical representations through multiple layers, making it particularly effective for speech and vision tasks
2. **Supervised learning** requires labeled data and involves minimizing a loss function through gradient descent
3. **Backpropagation** efficiently computes gradients using the chain rule, enabling training of deep networks
4. **Regularization techniques** (dropout, weight decay, batch normalization) help prevent overfitting
5. **Proper data splitting** (train/validation/test) is crucial for honest model evaluation
6. **Computational resources** (GPU memory, batch size, training time) must be carefully managed

```{admonition} Common Pitfalls to Avoid
:class: warning
- Training on the test set or using it for model selection
- Forgetting to normalize input data
- Using the same learning rate throughout training
- Not monitoring validation performance during training
- Ignoring computational constraints when designing models
```

With this foundation in mind, we are ready to deep dive into the architectures and applications of deep learning models for speech and vision.
