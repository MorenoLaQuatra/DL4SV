# Convolutional Neural Networks

```{figure} images/2_cnns/cover_cnn.png
---
width: 50%
name: cover
alt: cover
---
Image generated using [OpenDALL-E](https://huggingface.co/spaces/mrfakename/OpenDalleV1.1-GPU-Demo)
```

# Introduction

Convolutional Neural Networks (CNNs) are a class of deep neural networks that are widely used in computer vision applications. CNNs are designed to process data that have a grid-like or lattice-like topology, such as images, speech signals, and text.

The convolutional neural network architecture was first introduced in the 1980s by Yann LeCun and colleagues {cite:ps}`lecun1989backpropagation`. The first CNN was designed to recognize handwritten digits as the one shown in Figure {ref}`mnist`
. The network was trained on the MNIST dataset, a collection of 60000 handwritten digits. 

```{figure} images/2_cnns/mnist.png
---
width: 60%
name: mnist
alt: mnist
---
The MNIST dataset
```

## Properties of Image Data

Images are a special type of data that are characterized by three properties:
- **High dimensionality**: images are represented as a matrix of pixels, where each pixel is a value between 0 and 255. For example, a 224x224 RGB image is represented as a 224x224x3 matrix ($224 \times 224 \times 3 = 150528$).
- **Spatial correlation**: neaby pixel in an image are correlated. For example, in a picture of a cat, the pixels that represent the cat's fur are likely to be similar.
- **Invariance to geometric transformations**: the content of an image is invariant to geometric transformations such as translation, rotation, and scaling. For example, a picture of a cat is still a picture of a cat if we rotate it by 90 degrees.

Those properties are directly related to the fact that is really difficult to use a fully connected neural network to process images. 
- The high dimensionality of images makes the training of a fully connected neural network infeasible. Even a shallow network receiving as input a 224x224 RGB image would have 150528 input units in the first layer. This number would increase exponentially with the number of layers (2 layers $150528^2$, 3 layers $150528^3$, etc.).
- The spatial correlation of images is not exploited by fully connected neural networks. In a fully connected neural network, each input unit is connected to each output unit. This means that the network would learn a different weight for each pixel in the image, regardless of its position. This is not desirable because the network would not be able to learn the spatial correlation between pixels.
- Similarly, a fully connected neural network would not be able to learn the invariance to geometric transformations. If we translate, rotate, or scale an image, the network sees a completely different input. 

Convolutional Neural Networks are designed to overcome these limitations.

# Convolutional Neural Network Architecture

The architecture of a convolutional neural network is composed of three main components:
- **Convolutional layers**: these layers are responsible for extracting features from the input data. A convolutional layer is composed of a set of filters that are applied to the input data to extract features. The output of a convolutional layer is a set of feature maps, one for each filter.
- **Pooling layers**: these layers are responsible for reducing the dimensionality of the feature maps. A pooling layer is applied to each feature map independently. Average or max pooling are the most common pooling strategies.
- **Fully connected layers**: these layers are responsible for rearranging the features extracted by the convolutional layers into a vector of probabilities.

An example of a convolutional neural network architecture is shown in Figure {ref}`cnn_architecture`. The network is composed of five convolutional layers, five pooling layers, and tthree fully connected layers. The input of the network is a 224x224 RGB image. The output of the network is a vector of probabilities, one for each class in the dataset (i.e., 1000 in the case of [ImageNet](http://www.image-net.org/)).

```{figure} images/2_cnns/example_cnn.png
---
width: 80%
name: cnn_architecture
alt: cnn_architecture
---
Simple CNN architecture having all the basic components.
Image source [vitalflux.com](https://vitalflux.com/different-types-of-cnn-architectures-explained-examples/)
```

The convolution operation, together with the pooling operation, is the key component of a convolutional neural network. In the following sections, we will see how these operations work and how they can be used to extract features from images.

## Basic Components

Convolutional neural networks, also named ConvNets or CNNs, are a class of deep neural networks that have been initially designed to process images. The operations involved in a CNN are inspired by the visual cortex of the human brain. In particular, the convolution operation is inspired by the receptive fields of neurons in the visual cortex.

### Convolutional Layers

The main component of a CNN is the convolutional layer. In a convolutional layer, a set of filters is applied to the input data to extract **high-level features**. 

The convolution operation is defined by a set of parameters:
- The **filter size** is the size of the filter. The filter size is usually an odd number (e.g., 3x3, 5x5, 7x7). 
- **Stride** is the number of pixels by which the filter is shifted at each step.
- **Padding** is the number of pixels added to each side of the input image. Padding is usually used to preserve the spatial dimension of the input image.

The convolution operation is applied to each channel of the input image independently. For the moment, let's consider a single channel image. The convolution operation is defined as follows:
1. The filter is placed on the top-left corner of the input image.
2. The element-wise multiplication between the filter and the input image is computed.
3. The result of the multiplication is summed up to obtain a single value.
4. The filter is shifted by the stride value to the right. If the filter reaches the right side of the image, it is shifted to the left side of the next row.
5. Steps 2-4 are repeated until the filter reaches the bottom-right corner of the image.
6. The result of the convolution operation is a feature map, a matrix of values that represents the output of the convolution operation.

```{figure} images/2_cnns/images_cnn/conv_step0.png
---
width: 80%
name: convolution
alt: convolution
---
Initial settings of the convolution operation. 
On the left, the input image. On the middle, the filter. On the right, the output feature map (empty at the beginning). In the example, stride is 1 and padding is 0 (simpler case).
```

{numref}`convolution` shows the initial settings of the convolution operation. The feature map is empty at the beginning.

```{figure} images/2_cnns/images_cnn/conv_step1.png
---
width: 80%
name: conv_step_1
alt: convolution
---
Step 1 of the convolution operation. 
```

**Step 1**: the filter is placed on the top-left corner of the input image. The element-wise multiplication between the filter and the input image is computed. The result of the multiplication is summed up to obtain a single value. In this example, the result is 6, that is the first value of the feature map as shown in {numref}`conv_step_1`.

$$
\begin{align}
\begin{split}
& + 1 \times 1 + 0 \times 2 \\
& + 1 \times 5 + 0 \times 4 \\
& = 6
\end{split}
\end{align}
$$

```{figure} images/2_cnns/images_cnn/conv_step2.png
---
width: 80%
name: conv_step_2
alt: convolution
---
Step 2 of the convolution operation. 
```

**Step 2**: the filter is shifted by the stride value to the right. If the filter reaches the right side of the image, it is shifted to the left side of the next row. In this example, the filter is shifted by 1 pixel to the right. The result of the convolution operation is 8, that is the second value of the feature map as shown in {numref}`conv_step_2`.

```{figure} images/2_cnns/images_cnn/conv_stepn.png
---
width: 80%
name: conv_step_n
alt: convolution
---
Last step of the convolution operation.
```

**Step n**: after $n$ steps, the filter reaches the bottom-right corner of the image. The result of the convolution operation is a feature map, a matrix of values that represents the output of the convolution operation. In this example, the feature map is a 2x2 matrix as shown in {numref}`conv_step_n`.

The convolution operation is applied to each channel of the input image independently. The result is a set of feature maps, one for each channel of the input image. The number of feature maps is equal to the number of filters in the convolutional layer.

**Note**: in the previous example, we considered a *fixed* filter. In practice, the filters are *learned* during the training process. The weights of the filters are the parameters of the convolutional layer that the network learns during the training process.

### Pooling Layers

The pooling operation is often used to reduce the dimensionality of the feature maps. The pooling operation is defined by a set of parameters:
- The **pooling size** is the size of the pooling filter. As for convolutional kernels, the pooling size is usually an odd number (e.g., 3x3, 5x5, 7x7).
- **Stride** is the number of pixels by which the pooling filter is shifted at each step.
- **Padding** is the number of pixels added to each side of the input image. Padding is usually used to preserve the spatial dimension of the input image.
- **Pooling strategy** is the function used to aggregate the values in the pooling filter. The most common pooling strategies are average pooling and max pooling.

The pooling layer operates in a similar way to the convolutional layer. The pooling filter is placed on the top-left corner of the input image. The pooling operation is applied to each channel of the input image independently. The result of the pooling operation is a feature map, a matrix of values that represents the output of the pooling operation.

```{figure} images/2_cnns/images_cnn/avg_pooling.png
---
width: 60%
name: avg_pooling
alt: avg_pooling
---
Average pooling on a 4x4 matrix with pooling size 2x2 and stride 2. Padding is 0.
```

{numref}`avg_pooling` shows an example of average pooling. The colors help to visualize the pooling operation involving the following steps:

- The pooling filter is placed on the top-left corner of the input image. The average of the values in the pooling filter is computed. In this example, the average is $(1+2+5+6)/4 = 3.5$.
- The pooling filter is shifted by the stride value (2) to the right. Again, the average of the values in the pooling filter is computed. In this example, the average is $(3+4+7+8)/4 = 5.5$.
- Since we reached the right side of the image, the pooling filter is shifted to the left side of the next row. The average of the values in the pooling filter is computed. In this example, the average is $(9+10+13+14)/4 = 11.5$.
- ... the process continues until the pooling filter reaches the bottom-right corner of the image.

### Stride and Padding

To this point, we have seen that the convolution and pooling operations are defined by a set of parameters. In particular, the stride and padding parameters are used to control the spatial dimension of the output feature maps.

**Stride** is the number of elements (pixels in an image) by which the filter is shifted at each step. The stride parameter is used to control the spatial dimension of the output feature maps. 
Common values for the stride parameter are 1 and 2. A stride of 1 means that the filter is shifted by 1 pixel at each step. A stride of 2 means that the filter is shifted by 2 pixels at each step. 

```{figure} images/2_cnns/images_cnn/stride_step0.png
---
width: 80%
name: stride_step0
alt: stride
---
Step 1 of the convolution operation with stride 2. 
```

```{figure} images/2_cnns/images_cnn/stride_step1.png
---
width: 80%
name: stride_step1
alt: stride
---
Step 2 of the convolution operation with stride 2. 
```

```{figure} images/2_cnns/images_cnn/stride_step2.png
---
width: 80%
name: stride_step2
alt: stride
---
Step 3 of the convolution operation with stride 2. 
```

```{figure} images/2_cnns/images_cnn/stride_step3.png
---
width: 80%
name: stride_step3
alt: stride
---
Step 4 of the convolution operation with stride 2. 
```

{numref}`stride_step0` shows the initial settings of the convolution operation with stride 2. The filter is placed on the top-left corner of the input image. The result of the convolution operation is 6, that is the first value of the feature map. Similarly the process goes on until the filter reaches the bottom-right corner of the image. {numref}`stride_step1`, {numref}`stride_step2`, and {numref}`stride_step3` show the following steps of the convolution operation.

💡 Notice that the stride is the value by which the filter is shifted at each step **both along the horizontal and vertical dimensions**.

**Padding** is the number of elements (pixels in an image) added to each side of the input image. Padding is usually used to preserve the spatial dimension of the input image. 
In simple terms, padding is a "border" added to the input image. The value of the padding is usually 0 (zero padding - black border) or 1 (one padding - white border).

```{figure} images/2_cnns/images_cnn/padding.png
---
width: 80%
name: padding
alt: padding
---
Padding applied to a 4x4 matrix. Padding size is 1.
```

{numref}`padding` shows an example of padding. On the left, the input image. On the right, the padded image where the input matrix is reported in red and the padding is reported in blue. In this example, the padding size is 1.
With an input image of size $n \times n$, the output image has size $(n + 2p) \times (n + 2p)$, where $p$ is the padding size.
In the example of {numref}`padding`, the input image is a 4x4 matrix. The padding size is 1. The output image is a 6x6 matrix.

```{admonition} Computing the feature map size

Once set the parameters of the convolutional layer (filter size, stride, padding), it is possible to compute the size of the output feature map. The size of the output feature map is computed as follows:

$$
\begin{align}
\begin{split}
& \text{output size} = \frac{\text{input size} - \text{filter size} + 2 \times \text{padding}}{\text{stride}} + 1
\end{split}
\end{align}
$$

For example, if the input image is a 224x224 RGB image, the filter size is 3x3, the stride is 1, and the padding is 0, the size of the output feature map is:

$$
\begin{align}
\begin{split}
& \text{output size} = \frac{224 - 3 + 2 \times 0}{1} + 1 = 222
\end{split}
\end{align}
$$

Intuitively, the output feature map is smaller than the input image because the filter cannot be placed on the edges of the image. The **padding** is used to preserve the spatial dimension of the input image (+ sign in the equation). 
The **stride** instead is used to control the reduction of the spatial dimension of the output feature map (i.e., the denominator of the equation).

```

<!-- ### Activation Functions

The activation function is a non-linear function that is applied to the output of a layer. The activation function is usually applied after the convolutional and pooling layers. Similarly to fully connected neural networks, the activation function is used to introduce non-linearity in the network. The most common activation functions are:
- **ReLU** (Rectified Linear Unit): $f(x) = max(0, x)$
- **Sigmoid**: $f(x) = \frac{1}{1 + e^{-x}}$
- **Tanh**: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- **Softmax**: $f(x) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$

The *softmax* activation function is usually applied to the output of the last layer of the network. The softmax function is used to convert the output of the network into a vector of probabilities. The output of the network is a vector of values between 0 and 1. The sum of the values in the vector is equal to 1. Each value in the vector represents the probability that the input belongs to a specific class. -->

### Receptive Field

One relevant concept in convolutional neural networks is the **receptive field**. It is defined as the portion of the input image that is visible to a single neuron in the network. The receptive field is usually defined in terms of the number of pixels in the input image. For example, a receptive field of 3x3 means that the neuron can "see" a 3x3 portion of the input image.

```{figure} images/2_cnns/receptive_field.png
---
width: 80%
name: receptive_field
alt: receptive_field
---
Example of receptive field in a CNN.
```

{numref}`receptive_field` shows an example of receptive field in a CNN. The receptive field of the neuron in the third layer is 3x3 when considering the second layer. If we consider the input image (e.g., layer 1), the receptive field of the same neuron would be 9x9 (the entire image is 5x5 so the neuron can "see" the whole image).

Intuitively, the receptive field defines the region of the input image that has contributed to the activation of a neuron in the network.

## Common CNN Architectures

Since the introduction of the first CNN architecture in the 1980s, many different architectures have been proposed. They differ in terms of the number of layers, the number of filters, the pooling strategy, etc., but also in terms of architectural choices that have been made to improve both the performance and the training process of the network.

### LeNet-5

```{figure} images/2_cnns/lenet5.jpeg
---
width: 100%
name: lenet5
alt: lenet5
---
LeNet-5 architecture.
```

LeNet-5 {cite:ps}`lecun1989backpropagation` is the first CNN architecture proposed by Yann LeCun and colleagues. The network was designed to recognize handwritten digits. The network is composed of 7 layers: 3 convolutional layers, 2 pooling layers, and 2 fully connected layers. The input of the network is a 32x32 grayscale image. The output of the network is a vector of probabilities, one for each class in the dataset (i.e., 10 in the case of MNIST).

We can see that since the first convolutional layer, the number of filters (number of channels in the feature maps) increases from 1 (greyscale image) to 6. On the other hand, the spatial dimension of the feature maps decreases from 32x32 to 28x28. This is one of the main characteristics shared by many CNN architectures: going deeper in the network, the number of filters increases while the spatial dimension of the feature maps decreases.

### AlexNet

```{figure} images/2_cnns/images_cnn/alex_net.png
---
width: 100%
name: alex_net
alt: alex_net
---
AlexNet architecture.
```

AlexNet {cite:ps}`krizhevsky2012imagenet` is a CNN architecture proposed by Alex Krizhevsky and colleagues. The network was designed to classify images in the ImageNet dataset. The network is composed of 8 layers: 5 convolutional layers, 3 fully connected layers. The input of the network is a 224x224 RGB image. The output of the network is a vector of probabilities, one for each class in the dataset (i.e., 1000 in the case of ImageNet).

We can see that the network is composed of two groups of layers. The first group is composed of 5 convolutional layers and 3 pooling layers. The second group is composed of 3 fully connected layers. The first group is responsible for extracting features from the input image. The second group is responsible for classifying the input image.

It is worth mentioning that, at the time of its introduction, AlexNet was the first CNN architecture to use ReLU as activation function and dropout as regularization technique. AlexNet is also the winning entry of the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012, where it reduced the top-5 error by a large margin compared to the previous state-of-the-art.

### ResNet

```{figure} images/2_cnns/images_cnn/res_block.png
---
width: 30%
name: res_block
alt: res_block
---
ResBlock in ResNet architecture.
```

ResNet {cite:ps}`he2016deep` is a CNN architecture proposed by Kaiming He and colleagues. Similarly to AlexNet, the network was designed to classify images in the ImageNet dataset. This network is the first to introduce the concept of **residual learning** that allows training **very deep** networks. There are different versions of ResNet, with 50, 101, 152, and 200 layers. The network is composed of a concatenation of **residual blocks**. Each residual block is composed of two convolutional layers, batch normalization, ReLU activation function, and a shortcut connection (i.e., the residual connection).

{numref}`res_block` shows an example of residual block. The input of the residual block is a feature map. The feature map is processed by two convolutional layers. The output of the residual block is the sum of the input feature map and the output of the second convolutional layer. The residual block is composed of two convolutional layers and a shortcut connection. The shortcut connection is used to add the input feature map to the output of the second convolutional layer. The output of the residual block is the sum of the input feature map and the output of the second convolutional layer.

````{admonition} Residual Learning
:class: note, dropdown

Residual learning is a technique in training **exceptionally deep** neural networks. The fundamental concept involves incorporating a shortcut connection between the input and the output of a layer. Specifically, the output of the layer is formed by summing the input with the output of the layer.

The rationale behind residual learning lies in the belief that as layers are stacked in a network, the model can learn abstract features that are more useful for the downstream task than the shallower features acquired by less complex networks. However, this stacking of layers can bring to the **vanishing gradient problem**, a common obstacle in deep neural networks. The vanishing gradient problem manifests when the gradient of the loss function diminishes significantly with an increase in the number of layers. This slowdown in the gradient severely affect the training process.

Residual learning provides an elegant solution to the vanishing gradient problem by introducing a **shortcut connection** that directly links the input to the output of a layer. This shortcut connection facilitates an **uninterrupted flow** of the gradient from the output back to the input of the layer, effectively mitigating the challenges posed by the vanishing gradient problem. As a result, residual learning empowers the training of extremely deep networks, enabling them to capture intricate patterns and representations essential for complex tasks.

```{figure} images/2_cnns/residual_learning.gif
---
width: 80%
name: residual_learning
alt: residual_learning
---
Residual learning.
```

````

Many other CNN architectures have been proposed in the last years. Here are some references if the reader is interested in learning more about CNN architectures:
- VGG {cite:ps}`simonyan2014very`, introduced the concept of using small convolutional filters (3x3) with stride 1 and padding 1.
- DenseNet {cite:ps}`huang2017densely`, introduced the concept of dense blocks, where each layer is connected to all the previous layers.
- Inception {cite:ps}`szegedy2015going`, introduced the concept of inception modules, where the input is processed by different convolutional filters and the output is concatenated.
- MobileNet {cite:ps}`howard2017mobilenets`, introduced the concept of depthwise separable convolution, where the convolution operation is split into two separate operations: depthwise convolution and pointwise convolution.
- EfficientNet {cite:ps}`tan2019efficientnet`, introduced the concept of compound scaling, where the depth, width, and resolution of the network are scaled together.

# ConvNets for Audio Data

## Adapting CNNs for Sequential Data

## 1-D Convolutional Layers

### Convolutional Layers for Temporal Patterns

### Pooling Strategies

### Activation Functions

## Spectrogram Representation

### Short-Time Fourier Transform

### Mel-Spectrogram

## Audio CNN Architectures

### VGGish

### WaveNet [link](https://deepmind.google/discover/blog/wavenet-a-generative-model-for-raw-audio/)

### YAMNet [link](https://blog.tensorflow.org/2021/03/transfer-learning-for-audio-data-with-yamnet.html)


# Implementing a CNN with PyTorch

## Dataset

## Model

## Training

## Evaluation
