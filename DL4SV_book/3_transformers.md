# Transformers

```{figure} images/3_transformers/cover_transformers.png
---
width: 50%
name: cover
alt: cover
---
Image generated using [OpenDALL-E](https://huggingface.co/spaces/mrfakename/OpenDalleV1.1-GPU-Demo)
```

## Introduction

In this chapter, we will cover the basics of transformers, a type of neural network architecture that has been initially developed for natural language processing (NLP) tasks but has since been used and adapted for other modalities such as images, audio, and video.

### Why Transformers?

Before transformers, sequential data (text, audio, time series) was primarily processed using **Recurrent Neural Networks (RNNs)** and their variants like **LSTMs (Long Short-Term Memory)** and **GRUs (Gated Recurrent Units)**.

**Limitations of RNNs:**

1. **Sequential processing**: RNNs process data one step at a time, making them slow to train
   - Cannot parallelize computation across the sequence
   - Training time grows linearly with sequence length

2. **Long-range dependencies**: Despite LSTMs and GRUs, modeling very long sequences remains challenging
   - Information from early tokens gets diluted as the sequence progresses
   - Gradient vanishing/exploding still occurs for very long sequences

3. **Fixed context**: Each position can only access information from previous positions (in standard RNNs)
   - Cannot look ahead in the sequence
   - Cannot directly access arbitrary positions

**The Transformer Solution:**

Transformers address these limitations through the **attention mechanism**:
- **Parallel processing**: All positions are processed simultaneously
- **Direct access**: Each position can directly attend to any other position
- **Flexible context**: Can look at the entire sequence (past and future) at once
- **Scalability**: Easily scale to very long sequences with efficient implementations

**Impact:** Since their introduction in 2017, transformers have become the dominant architecture in:
- Natural Language Processing (BERT, GPT, T5)
- Computer Vision (ViT, DETR)
- Speech Processing (Wav2Vec 2.0, Whisper)
- Multimodal AI (CLIP, DALL-E)

The transformer architecture is designed for modeling sequential data, such as text, audio, and video. It is based on the idea of self-attention, which is a mechanism that allows the network to learn the relationships between different elements of a sequence. For example, in the case of an audio sequence, the network can learn the relationships between different frames of the audio signal and leverage the correlations between them to perform a task such as speech recognition.

The original Transformer architecture was introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) {cite:ps}`vaswani2017attention`
by Vaswani et al. in 2017. Since then, many variants of the original architecture have been proposed, and transformers have become the state-of-the-art architecture for many tasks. In this chapter, we will cover all the building blocks of the transformer architecture and show how they can be used for different tasks.

````{admonition} Deep Dive: Understanding RNNs, LSTMs, and GRUs
:class: dropdown

To fully appreciate why transformers are revolutionary, we need to understand the architectures they replaced. **Recurrent Neural Networks (RNNs)** were the dominant approach for sequential data before transformers.

### Basic Recurrent Neural Networks

An RNN processes a sequence one element at a time, maintaining a **hidden state** that captures information about the sequence seen so far.

**Architecture:**

At each time step $t$, an RNN takes:
- Current input $x_t$
- Previous hidden state $h_{t-1}$

And produces:
- New hidden state $h_t$
- Output $y_t$ (optional)

The computation is:

$$
\begin{align}
h_t &= \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h) \\
y_t &= W_{hy} h_t + b_y
\end{align}
$$

where $W_{hh}$, $W_{xh}$, and $W_{hy}$ are weight matrices, and $b_h$, $b_y$ are bias vectors.

```{figure} images/3_transformers/rnn_unrolled.png
---
width: 100%
name: rnn_unrolled
alt: rnn_unrolled
---
RNN unrolled through time. The same weights are shared across all time steps.
```

**Key insight:** The same weights are applied at every time step. The network "remembers" previous inputs through the hidden state, which is updated at each step.

**Simple PyTorch implementation:**

```{code-block} python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Weights
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        
    def forward(self, x, hidden):
        # x: (batch_size, input_size)
        # hidden: (batch_size, hidden_size)
        
        # Concatenate input and hidden state
        combined = torch.cat([x, hidden], dim=1)
        
        # Compute new hidden state
        hidden = torch.tanh(self.i2h(combined))
        
        # Compute output
        output = self.i2o(combined)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# Process a sequence
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
sequence_length = 15
batch_size = 32

hidden = rnn.init_hidden(batch_size)
for t in range(sequence_length):
    x_t = torch.randn(batch_size, 10)  # Input at time t
    output, hidden = rnn(x_t, hidden)
```

### The Vanishing Gradient Problem

While simple RNNs work in theory, they suffer from the **vanishing gradient problem** when training on long sequences.

**Why it happens:**

During backpropagation through time (BPTT), gradients are propagated backward through the sequence. At each step, they are multiplied by the weight matrix and the derivative of the activation function:

$$
\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}
$$

For long sequences (large $T$):
- If the product terms are < 1, gradients **vanish** (approach zero)
- If the product terms are > 1, gradients **explode** (become very large)

**Consequence:** The network cannot learn long-range dependencies. Information from time step 1 is lost by time step 100.

**Example:** In the sentence "The cat, which we found yesterday in the garden, **was** hungry", the RNN must remember "cat" (singular) over many words to correctly predict "was" (not "were").

### Long Short-Term Memory (LSTM)

LSTMs {cite:ps}`hochreiter1997long` were designed to solve the vanishing gradient problem by introducing a **memory cell** and **gating mechanisms**.

**Key innovation:** Instead of just a hidden state, LSTMs maintain:
- **Cell state** $c_t$: Long-term memory (flows through time with minimal modifications)
- **Hidden state** $h_t$: Short-term memory (output of the LSTM)

**Three gates control information flow:**

1. **Forget gate** ($f_t$): What information to discard from cell state
2. **Input gate** ($i_t$): What new information to add to cell state
3. **Output gate** ($o_t$): What information to output from cell state

**LSTM equations:**

$$
\begin{align}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(input gate)} \\
\tilde{c}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad \text{(candidate values)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(update cell state)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(output gate)} \\
h_t &= o_t \odot \tanh(c_t) \quad \text{(hidden state)}
\end{align}
$$

where $\sigma$ is the sigmoid function, $\odot$ is element-wise multiplication, and $[\cdot, \cdot]$ denotes concatenation.

```{figure} images/3_transformers/lstm_cell.jpg
---
width: 100%
name: lstm_cell
alt: lstm_cell
---
LSTM cell showing the three gates and cell state flow. The cell state (top horizontal line) can flow through time with minimal modifications.
```

**How it solves vanishing gradients:**

The cell state $c_t$ acts as a "highway" for gradients to flow backward through time. The forget gate can learn to keep important information in the cell state for many time steps without modification, allowing gradients to flow without vanishing.

**Intuition through an example:**

Consider processing the sentence "The cat sat on the mat":
- **Forget gate**: "Should I forget that we're talking about a cat?" → No (keep it)
- **Input gate**: "Should I add new information about 'sat'?" → Yes (add action)
- **Output gate**: "Should I output information about the cat?" → Yes (for next word prediction)

**PyTorch implementation:**

```{code-block} python
import torch
import torch.nn as nn

# Built-in LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)

# Process sequence
batch_size = 32
sequence_length = 15
input_size = 10

x = torch.randn(batch_size, sequence_length, input_size)
output, (h_n, c_n) = lstm(x)

# output: (batch_size, sequence_length, hidden_size) - outputs at each time step
# h_n: (num_layers, batch_size, hidden_size) - final hidden state
# c_n: (num_layers, batch_size, hidden_size) - final cell state
```

### Gated Recurrent Unit (GRU)

GRUs {cite:ps}`cho2014learning` are a simplified variant of LSTMs with fewer parameters but similar performance.

**Key differences from LSTM:**
- **No separate cell state**: Only hidden state $h_t$
- **Two gates instead of three**: Update gate and reset gate
- **Fewer parameters**: ~25% fewer than LSTM

**GRU equations:**

$$
\begin{align}
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t]) \quad \text{(update gate)} \\
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t]) \quad \text{(reset gate)} \\
\tilde{h}_t &= \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) \quad \text{(candidate hidden state)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(final hidden state)}
\end{align}
$$

**Gate functions:**
- **Reset gate** ($r_t$): Controls how much of the previous hidden state to use when computing the candidate hidden state
- **Update gate** ($z_t$): Controls how much of the previous hidden state to keep and how much of the candidate to use

```{figure} images/3_transformers/gru_cell.ppm
---
width: 100%
name: gru_cell
alt: gru_cell
---
GRU cell showing the two gates. Simpler than LSTM but often comparable performance.
```

**PyTorch implementation:**

```{code-block} python
import torch
import torch.nn as nn

# Built-in GRU
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)

x = torch.randn(32, 15, 10)  # (batch, sequence, features)
output, h_n = gru(x)
```

### LSTM vs GRU: When to Use Which?

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Parameters** | More (4 weight matrices) | Fewer (3 weight matrices) |
| **Training speed** | Slower | Faster (fewer parameters) |
| **Memory** | Higher (stores cell state) | Lower |
| **Performance** | Slightly better on complex tasks | Comparable on most tasks |
| **Long sequences** | Better at very long dependencies | Good for moderate-length sequences |

**Rule of thumb:**
- Start with **GRU** (simpler, faster, fewer hyperparameters)
- Switch to **LSTM** if you need to model very long-range dependencies
- In practice, the difference is often small

### Why Transformers Replaced RNNs/LSTMs/GRUs

Despite solving the vanishing gradient problem, LSTMs and GRUs still have fundamental limitations:

**1. Sequential Processing Bottleneck**
```
RNN/LSTM/GRU: h₁ → h₂ → h₃ → h₄ → h₅
               ↓    ↓    ↓    ↓    ↓
              Cannot parallelize!

Transformer:   All positions processed simultaneously
               ↓
              Full parallelization!
```

**2. Limited Context Window**

Even with LSTMs/GRUs, information from very distant positions is diluted. A 1000-word document requires 1000 sequential steps, with information potentially degrading over that distance.

**3. Difficulty in Learning Which Positions Matter**

RNNs treat all previous positions equally (through the hidden state). Transformers use attention to explicitly learn which positions are important for each prediction.

**Comparison Summary:**

| Feature | RNN/LSTM/GRU | Transformer |
|---------|--------------|-------------|
| **Parallelization** | No (sequential) | Yes (all positions at once) |
| **Training speed** | Slow for long sequences | Fast (with sufficient hardware) |
| **Long-range dependencies** | Challenging despite gates | Natural through attention |
| **Interpretability** | Hidden state (opaque) | Attention weights (interpretable) |
| **Memory complexity** | O(n) | O(n²) (due to attention) |

**When RNNs are still relevant:**
- Very long sequences where O(n²) attention is prohibitive
- Streaming applications (processing unbounded sequences online)
- Extremely limited computational resources
- Specific domains where sequential inductive bias helps

However, for most modern applications, transformers have become the default choice due to their superior performance and parallelization capabilities.

**Images Needed for This Section:**

To fully illustrate the RNN concepts, please add the following images to `images/3_transformers/`:

1. **rnn_unrolled.png**: Diagram showing an RNN unrolled through time (3-4 time steps), with the same weights $W$ applied at each step, showing how $h_t$ flows through time
2. **lstm_cell.png**: Detailed LSTM cell diagram showing the cell state (horizontal line at top), three gates (forget, input, output) with sigmoid activations, tanh activations, and element-wise operations (×, +)
3. **gru_cell.png**: GRU cell diagram showing update gate and reset gate, how they control information flow, and the simpler structure compared to LSTM

You can find good reference diagrams at [Colah's blog on LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) or create simple diagrams showing the flow of information.

````

```{figure} images/3_transformers/architecture.webp
---
width: 50%
name: architecture
alt: architecture
---
Transformer architecture
```

# Architecture Overview

{numref}`architecture` shows the architecture of the transformer model. The model consists of an encoder and a decoder. 

- The **encoder** is responsible for processing the input sequence and extracting relevant information.
- The **decoder** is responsible for generating the output sequence based on the information extracted by the encoder.

The encoder and decoder are composed of a stack of identical layers. Each layer consists of multiple components, including a multi-head self-attention mechanism and a feed-forward network. We will cover each of these components in detail in the following sections.

💡 Transformers are designed to model **discrete** sequences, such as words in text, genes in DNA, or tokens in a programming language. They are not designed to model **continuous** sequences, such as audio or video. However, transformers can be used to model continuous sequences by discretizing them with specific techniques. 

A few concepts are important to understand before we dive into the details of the transformer architecture.

- **Pre-training and fine-tuning** is a technique that consists of training a model on a large amount of unlabeled data. The model is then fine-tuned on a specific task using a small amount of labeled data. Pre-training is a common technique used in deep learning to improve the performance of a model on a specific task. BERT {cite:ps}`devlin2018bert`, Wav2Vec 2.0 {cite:ps}`baevski2020wav2vec`, and ViT {cite:ps}`dosovitskiy2020image` are examples of models that have been pre-trained on large amounts of data and fine-tuned on specific tasks.
- **Discretization** is used to convert continuous sequences into discrete sequences. For example, an audio signal is a continuous sequence, to convert it into a discrete sequence, we can split it into frames and define a *vocabulary* of frames. Once discretized, we can use *classification-like* approaches to train a transformer model on the audio sequence. Wav2Vec 2.0 {cite:ps}`baevski2020wav2vec` is an example of a model that uses discretization to train a transformer model on audio sequences.
- **Positional encoding** is a technique that consists of injecting information about the position of each element of a sequence into the model. We will see later that *attention* is a mechanism that allows the model to learn the relationships between the different elements of a sequence but it does not take into account the position of each element. Positional encoding is used to inject this information into the model.
- **Encoder models** are transformer models that only have an encoder. They are used to extract features from a sequence. BERT {cite:ps}`devlin2018bert` and ViT {cite:ps}`dosovitskiy2020image` are examples of encoder models.
- **Decoder models** are transformer models that only have a decoder. They are used to generate a sequence based on a set of features. GPT-2 {cite:ps}`radford2019language` and VioLA {cite:ps}`wang2023viola` are examples of decoder models.
- **Sequence-to-sequence models** are transformer models that have both an encoder and a decoder. They are used to generate a sequence based on another sequence. BART {cite:ps}`lewis2019bart` and Whisper {cite:ps}`radford2023robust` are examples of sequence-to-sequence models.

Those concepts will be used throughout this chapter to describe the different transformer models.

## Encoder

The encoder is responsible for processing the input sequence and extracting relevant information. The goal is to train a neural network that can leverage the correlations between the different elements of the input sequence to perform **discriminative** tasks such as classification, regression, or sequence labeling.

The input of the encoder is a sequence of elements. For example, in the case of text, the input sequence is a sequence of words. In the case of audio, the input sequence is a sequence of frames. In the case of images, the input sequence is a sequence of patches. The sequence is first converted into a sequence of *vector embeddings* that are then processed by the encoder layers.

```{figure} images/3_transformers/encoder_with_tensors_2.png
---
width: 70%
name: encoder
alt: encoder
---
Encoder Layer architecture. Image source [illustrated-transformer](http://jalammar.github.io/illustrated-transformer/)
```

{numref}`encoder` shows the architecture of an encoder layer. The input sequence is first converted into a sequence of vector embeddings $X = \{x_1, x_2, ..., x_n\}$ using an embedding layer. The embeddings are then processed by the self-attention layer and then pass through a feed-forward network. The output of the feed-forward network is then added to the input embeddings to produce the output embeddings $Y = \{y_1, y_2, ..., y_n\}$.

The encoder is composed of a stack of identical layers, all similar to the one shown in {numref}`encoder`. The output of the encoder is the output embeddings $Y$ of the last layer.

## Decoder

The decoder is responsible for generating the output sequence based on the information extracted by the encoder. The goal is to train a neural network that can leverage the correlations between the different elements of the input sequence to perform **generative** tasks such as text generation, image generation, or speech synthesis.

The input of the decoder is a sequence of elements. For example, in the case of audio, the input sequence is a sequence of frames. The sequence is first converted into a sequence of *vector embeddings* that are then processed by the decoder layers.

```{figure} images/3_transformers/transformer-decoder-intro.png
---
width: 70%
name: decoder
alt: decoder
---
Decoder Layer architecture. Image source [illustrated-gpt-2](https://jalammar.github.io/illustrated-gpt2/)
```

The *masked self-attention* layer is similar to the self-attention layer of the encoder. The only difference is that the masked self-attention layer is masked to prevent the decoder from "seeing" the future elements of the sequence. The output is then processed by a feed-forward network. The output of the feed-forward network is then added to the input embeddings to produce the output embeddings $Y = \{y_1, y_2, ..., y_n\}$.

When training a decoder-only model, the output embeddings $Y$ at each position $i$ are used to predict the next element of the sequence $y_{i+1}$. 

💡 In contrast with RNNs, the training of the decoder is **autoregressive**. This means that the model is trained to predict the next element of the sequence based on the previous elements of the sequence. While for RNNs we need to recursively feed the output of the model back as input, for transformers we can compute the output of the model in parallel for all the elements of the sequence.

💡 During **inference**, the decoder is used to generate the output sequence. However, the target is not available during inference. Instead, the output of the decoder at each position $i$ is used as input for the next position $i+1$. This process is repeated until a special token is generated or a maximum number of steps is reached. This is one of the reason why, the **inference** on transformers is slower than the **training**.

## Encoder-Decoder

If we combine the encoder and decoder, we get a sequence-to-sequence model. The encoder is used to extract features from the input sequence and the decoder is used to generate the output sequence based (or conditioned) on the extracted features.
One example of sequence-to-sequence model is a music style transfer model. The input sequence may be a song in a specific style and the output sequence may be the same song in another style. The encoder is used to extract features from the input song and the decoder is used to generate the output song based on the extracted features.

```{figure} images/3_transformers/encoder_decoder.png
---
width: 70%
name: encoder_decoder
alt: encoder_decoder
---
Encoder-Decoder architecture. Image source [illustrated-transformer](http://jalammar.github.io/illustrated-transformer/)
```

{numref}`encoder_decoder` shows the architecture of an encoder-decoder model. The input sequence is first converted into a sequence of vector embeddings $X = \{x_1, x_2, ..., x_n\}$ using an embedding layer. The embeddings are then processed by the encoder layers. The output of the encoder is the output embeddings $Y = \{y_1, y_2, ..., y_n\}$ of the last layer. The output embeddings are then processed by the decoder layers. Here we have an **additional** attention layer that allows the decoder to combine the output embeddings of the encoder with the output embeddings of the decoder. The output of the decoder is the output embeddings $Z = \{z_1, z_2, ..., z_n\}$ of the last layer.

🖊️ The encoder-decoder attention is usually referred to as the **cross-attention** layer. The self-attention layer in the encoder is usually referred to as the **self-attention** layer. The self-attention layer in the decoder is usually referred to as the **masked self-attention** layer because it is masked to prevent the decoder from "seeing" the future elements of the sequence. All these layers, however, perform the same operation that we will describe in the following sections.



# Transformer Components

```{figure} images/3_transformers/architecture.webp
---
width: 50%
name: architecture_2
alt: architecture_2
---
Encoder-decoder Transformer architecture.
```

We will describe the different components of the transformer architecture from **bottom to top**. We will follow the {numref}`architecture` and start with the embedding layer, then the positional encoding, the self-attention layer, and so on.

## Embedding Layer

When training a transformer model, the input sequence is first converted into a sequence of *vector embeddings*. Those vector embeddings are created using an embedding layer. The embedding layer is a simple linear layer that maps each element of the input sequence to a vector of a specific size. The size of the vector is called the *embedding size* and is a power of 2, usually between 128 and 1024. The embedding size is a **hyperparameter** of the model.

We can see the embedding layer as a lookup table that maps each element of the input sequence to a vector of a specific size. The embedding layer is initialized randomly and is trained through backpropagation. The embedding layer is usually the first layer of the encoder and the decoder.

```{figure} images/3_transformers/lookup_table.gif
---
width: 70%
name: lookup_table
alt: lookup_table
---
Embedding layer as a lookup table. Image source [lena-voita](https://lena-voita.github.io/nlp_course/word_embeddings.html)
```

{numref}`lookup_table` shows an example of an embedding layer in the context of NLP (it is simpler to visualize in this context). The embedding layer is a lookup table that maps each word of the input sequence to a vector of a specific size.

(pos_encoding_section)=
## Positional Encoding

After the embedding layer, the input sequence is converted into a sequence of vector embeddings. As we can see later, the attention mechanism, at the core of the transformer architecture, does not take into account the position of each element of the sequence. To inject this information into the model, we use a technique called *positional encoding*.

### Why is Positional Encoding Needed?

The attention mechanism treats the input as a **set**, not a **sequence**. If we shuffle the input tokens, the attention output remains the same (ignoring the learned parameters). However, for most tasks, **order matters**:
- In text: "dog bites man" vs "man bites dog"
- In audio: the temporal order of sounds
- In video: the sequence of frames

Positional encoding solves this by adding position information to each token's embedding, making the input order-aware.

### Sinusoidal Positional Encoding

There are different implementations of positional encoding. The traditional implementation is based on sinusoidal functions. For each position $i$ of the input sequence, we compute a vector $PE_i$ of the same size as the embeddings. The vector $PE_i$ is then added to the embeddings $x_i$ to produce the final embeddings $x_i + PE_i$.

**How can we compute the vector $PE_i$?** The vector $PE_i$ is computed using a combination of sinusoidal functions. We define a set of frequencies $f$ and compute the vector $PE_i$ as follows:

$$PE_i = \begin{bmatrix} sin(f_1 \times i) \\ cos(f_1 \times i) \\ sin(f_2 \times i) \\ cos(f_2 \times i) \\ \vdots \\ sin(f_{d/2} \times i) \\ cos(f_{d/2} \times i) \end{bmatrix}$$

where $d$ is the size of the embeddings. The frequencies $f$ are computed as follows:

$$f_i = \frac{1}{10000^{2i/d}}$$

The frequencies $f$ are computed using a geometric progression. The first frequency is $f_1 = 1/10000^{2 \times 1/d}$, the second frequency is $f_2 = 1/10000^{2 \times 2/d}$, and so on. The frequencies are then used to compute the vector $PE_i$. $d$ is the size of the embeddings. The vector $PE_i$ is then added to the embeddings $x_i$ to produce the vector $x_i + PE_i$ that will be the input of the network.

**Why sinusoidal functions?**

1. **Unique encoding**: Different positions get different patterns of sine and cosine values
2. **Smooth transitions**: Nearby positions have similar encodings
3. **Extrapolation**: The model can potentially generalize to sequence lengths not seen during training
4. **Relative positions**: For any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$, making it easier for the model to learn relative positions

**Alternative: Learned Positional Embeddings**

Instead of fixed sinusoidal encodings, some models learn positional embeddings:
```python
self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)
```

This approach:
- Can learn task-specific position representations
- Limited to sequences shorter than or equal to max_seq_length seen during training
- Used in BERT and many other models


```{figure} images/3_transformers/transformer_positional_encoding_example.png
---
width: 100%
name: positional_encoding
alt: positional_encoding
---
Positional encoding example. Image source [illustrated-transformer](http://jalammar.github.io/illustrated-transformer/)
```

{numref}`positional_encoding` shows an example of positional encoding (again in the context of NLP). Each element in a $PE_i$ vector is computed using a sinusoidal function.

💡 The idea behind the use of sinusoidal functions is to allow the model to be able to encode regularities in the position of the elements of the sequence. Different frequencies may have similar values with different regularities. In a complete data-driven approach, the model would learn the regularities according to the patterns found in the data.

## Attention Mechanism

At this point of the chapter, we have converted the input sequence into a sequence of vector embeddings. The next step is to process the embeddings using the attention mechanism. The attention mechanism is the core of the transformer architecture. It is used to learn the relationships between the different elements of the sequence.

### Understanding Attention: A Concrete Example

Before diving into the mathematical formulation, let us understand attention with a simple numerical example. Consider a sequence of 3 words with embedding size 4:

Input embeddings (simplified for illustration):
```
Word 1: [1.0, 0.5, 0.2, 0.8]
Word 2: [0.3, 0.9, 0.6, 0.4]
Word 3: [0.7, 0.2, 0.9, 0.3]
```

For each word, we want to compute a new representation that considers information from all other words. Let us focus on computing the new representation for Word 1.

**Step 1: Compute Query, Key, Value**

For simplicity, assume our learned linear transformations are identity matrices (in practice, these are learned):
```
Q1 (query for word 1) = [1.0, 0.5, 0.2, 0.8]
K1 (key for word 1) = [1.0, 0.5, 0.2, 0.8]
K2 (key for word 2) = [0.3, 0.9, 0.6, 0.4]
K3 (key for word 3) = [0.7, 0.2, 0.9, 0.3]

V1 (value for word 1) = [1.0, 0.5, 0.2, 0.8]
V2 (value for word 2) = [0.3, 0.9, 0.6, 0.4]
V3 (value for word 3) = [0.7, 0.2, 0.9, 0.3]
```

**Step 2: Compute Attention Scores**

Compute dot product between Q1 and all keys:
```
Score(Q1, K1) = 1.0*1.0 + 0.5*0.5 + 0.2*0.2 + 0.8*0.8 = 1.93
Score(Q1, K2) = 1.0*0.3 + 0.5*0.9 + 0.2*0.6 + 0.8*0.4 = 1.19
Score(Q1, K3) = 1.0*0.7 + 0.5*0.2 + 0.2*0.9 + 0.8*0.3 = 1.22
```

**Step 3: Apply Softmax**

Normalize scores to get attention weights (scaled by 1/√4 = 0.5 in practice):
```
Before softmax: [1.93, 1.19, 1.22]
After softmax: [0.51, 0.24, 0.25]  (approximately)
```

Interpretation: Word 1 should pay 51% attention to itself, 24% to Word 2, and 25% to Word 3.

**Step 4: Compute Weighted Sum**

Combine values using attention weights:
```
Output1 = 0.51 * V1 + 0.24 * V2 + 0.25 * V3
        = 0.51 * [1.0, 0.5, 0.2, 0.8] + 0.24 * [0.3, 0.9, 0.6, 0.4] + 0.25 * [0.7, 0.2, 0.9, 0.3]
        = [0.76, 0.52, 0.47, 0.65]
```

The new representation for Word 1 is a weighted combination of all words, where the weights reflect their relevance.

### Formal Definition

The attention mechanism is a mechanism that allows the model to learn the relationships between the different elements of the sequence. the process can be divided into three steps:
1. **Query, Key, and Value**. The input embeddings are first *split* into three vectors: the query vector, the key vector, and the value vector. 
2. **Attention**. The query vector is compared to the key vector to produce a score, e.g., a float value between 0 and 1. The score is then used to compute a weighted average of the value vector. The weighted average is called the *attention vector*.
3. **Output**. The attention vector is then processed by a linear layer to produce the output vector.

```{figure} images/3_transformers/attention.gif
---
width: 100%
name: attention_animation
alt: attention_animation
---
Attention mechanism steps. Image source [towardsdatascience](https://towardsdatascience.com/illustrated-self-attention-2d627e33b2)
```

{numref}`attention_animation` shows an example of the attention mechanism. The input embeddings are first split into three vectors: the query vector, the key vector, and the value vector.
- **Query**. The query vector is used to *ask* to all other elements of the sequence *how much* they are related to the current element. The dot product between the query vector and the key vector is used to compute a score (the higher the score, the more related the elements are). The score is then normalized using a softmax function to produce a probability distribution over all the elements of the sequence.
- **Key**. The key vector is used to *answer* to the query. Each value vector is multiplied with the *query* of the other elements of the sequence. *query* and *key* are the vectors that, multiplied together, produce the score.
- **Value**. Once obtained a score for each element of the sequence, the *value* vector is multiplied with the score to produce a weighted average of the value vector. The weighted average is called the *attention vector*.

The final vector representation for a given input element of the sequence is given by the sum of the attention vectors of all the elements of the sequence. 

💡 Note that, the attention mechanism is usually referred to *self*-attention because the attention score is computed between a given element of the sequence and all the other elements, including itself.

If we put this into equations, we have:

$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

where $Q$ is the query vector, $K$ is the key vector, $V$ is the value vector, and $d_k$ is the size of the key vector. $\sqrt{d_k}$ is used to scale the dot product between $Q$ and $K$. The softmax function is applied to each row of the matrix $QK^T$ to produce a probability distribution over all the elements of the sequence. The probability distribution is then used to compute a weighted average of the value vector $V$.
The formula above is the *matrix form* of the attention mechanism.

**Why scale by √d_k?** Without scaling, for large embedding dimensions, the dot products can become very large, pushing the softmax into regions with extremely small gradients. Scaling by $\sqrt{d_k}$ keeps the variance of the dot products stable regardless of the embedding dimension, ensuring better gradient flow during training.

**Computational complexity:** For a sequence of length $n$ and embedding dimension $d$:
- Computing $QK^T$: $O(n^2 d)$ - quadratic in sequence length
- This is the main bottleneck for very long sequences
- Various efficient attention mechanisms (sparse attention, linear attention) have been proposed to address this

Each element of the sequence is processed independently by the attention mechanism. This means that the attention mechanism can be computed in parallel for all the elements of the sequence. This is one of the reasons why transformers are faster than RNNs on modern hardware (e.g., GPUs).

Coming back to our bottom-up approach, the attention mechanism is used to process the embeddings of the input sequence (after the positional encoding). The output of the attention mechanism is a sequence of vectors having the same size and shape as the input embeddings. After the attention mechanism a simple linear layer is used to produce the output embeddings (typically without altering the size of the embeddings).

**Multi-head Attention**. The attention mechanism described above is called *single-head attention*. In practice, the attention mechanism is computed multiple times in parallel on subsets of the embeddings. Each subset is called a *head*. Before feeding the embedding into the self-attention layer, in case of multi-head attention, the vector is first split into parts and each part is processed by a different head. The output of the self-attention layer is then the concatenation of the output of each head. The output of the self-attention layer is then processed by a linear layer to produce the output embeddings.

```{figure} images/3_transformers/multi-head-attention.svg
---
width: 50%
name: multi-head-attention
alt: multi-head-attention
---
Example of multi-head attention (e.g., number of attention heads $h=2$).
```

{numref}`multi-head-attention` shows an example of multi-head attention. In practice, all implementations of modern transformer models use multi-head attention (e.g., BERT, GPT-2, ViT, etc.). The number of heads is a **hyperparameter** of the model, similarly to the embedding size. It is worth noting that, the number of heads should be a number such that the size of the input embedding is divisible by the number of heads.
For example, if the embedding size is $512$, we can use $8$ heads ($512/8 = 64$) or $16$ heads ($512/16 = 32$) but not $10$ heads ($512/10 = 51.2$).

## Feed-Forward and Residual Connections

After the attention mechanism, the output embeddings are processed by a feed-forward network. The feed-forward network is a simple linear layer followed by a non-linear activation function (e.g., ReLU). 

Similarly to what we have seen with ResNets {cite:ps}`he2015deep`, in each layer of the encoder and decoder, there are *residual connections* that sum up the output of a sub-layer with the input of the sub-layer.

```{figure} images/3_transformers/transformer_residual_layer_norm_2.png
---
width: 50%
name: transformer_residual_layer_norm
alt: transformer_residual_layer_norm
---
Residual connections and layer normalization. Image source [illustrated-transformer](http://jalammar.github.io/illustrated-transformer/)
```

{numref}`transformer_residual_layer_norm` shows an example of residual connections and layer normalization. Both the output of the attention layer and the output of the feed-forward network are added to the input embeddings. The output of the residual connections is then processed by a layer normalization layer. Layer normalization is a technique that consists of normalizing the output of a layer using the mean and variance of the output of the layer. Layer normalization is used to make the training of the model more stable and efficient.

## Encoder and Decoder Models

At this point we have all the ingredients to create an encoder transformer layer. The encoder model is composed of:
- **Embedding layer**. The embedding layer converts the input sequence into a sequence of vector embeddings.
- **Positional encoding**. The positional encoding injects information about the position of each element of the sequence into the model.
- **Encoder layers**. The encoder layers process the embeddings using:
    - **Multi-head attention**. The multi-head attention mechanism is used to learn the relationships between the different elements of the sequence.
    - **Feed-forward network**. The feed-forward network is used to process the output of the multi-head attention mechanism.
    - **Residual connections**. The residual connections are used to sum up the output of the multi-head attention mechanism with the input embeddings.
    - **Layer normalization**. The layer normalization is used to normalize the output of the encoder layer.

A stack of encoder layers is used to create the encoder model. The output of the encoder is the output embeddings of the last encoder layer.
We can use this encoder model to extract features from a sequence. For example, we can use the encoder model to extract features from an audio sequence and then use those features to perform speech recognition.

The decoder model, when used in decoder-only mode, is similar to the encoder model. The decoder model is composed of:
- **Embedding layer**. The embedding layer converts the input sequence into a sequence of vector embeddings.
- **Positional encoding**. The positional encoding injects information about the position of each element of the sequence into the model.
- **Decoder layers**. The decoder layers process the embeddings using:
    - **Masked multi-head attention**. The masked multi-head attention mechanism is used to learn the relationships between the different elements of the sequence. The attention mechanism is masked to prevent the decoder from "seeing" the future elements of the sequence.
    - **Feed-forward network**. The feed-forward network is used to process the output of the multi-head attention mechanism.
    - **Residual connections**. The residual connections are used to sum up the output of the multi-head attention mechanism with the input embeddings.
    - **Layer normalization**. The layer normalization is used to normalize the output of the decoder layer.

Notice that, the attention layer of the decoder is referred as **masked** multi-head attention because it is masked to prevent the decoder from "seeing" the future elements of the sequence.

```{figure} images/3_transformers/masked-self-attention.png
---
width: 75%
name: masked-self-attention
alt: masked-self-attention
---
Comparison between self-attention and masked self-attention. Image source [illustrated-transformer](http://jalammar.github.io/illustrated-transformer/)
```

{numref}`masked-self-attention` shows an example of masked self-attention and the comparison with self-attention operation. When processing element $x_i$, the masked self-attention mechanism is masked to prevent the decoder from "seeing" the future elements of the sequence $x_{i+1}, x_{i+2}, ..., x_n$. The *masked*-self attention mechanism is usually implemented by adding a mask to the softmax function. The mask contains $-\infty$ values for all the elements of the sequence that we want to mask. The $-\infty$ values are used to set the attention score to $0$ after the softmax function. This means that the decoder will not be able to attend to the masked elements of the sequence.

## Encoder-Decoder (Sequence-to-sequence) Models

We have seen how to design layers for the encoder and decoder models. **Encoder** models are used to extract features from a sequence. **Decoder** models are used to generate a sequence *based on* the previous elements of the sequence. **Encoder-decoder** models are used to generate data *conditioned on* another sequence. For example, we can use an encoder-decoder model to translate a sentence from English to French or to generate the transcription of an audio sequence.

The encoder-decoder model is composed of:
- **Encoder**. The encoder is used to extract features from the input sequence.
- **Decoder**. The decoder is used to generate the output sequence based on the extracted features.
  - **Cross-attention (encoder-decoder attention)**. The decoder in this case adds an additional attention layer that allows the decoder to *condition* the output sequence on the input sequence. The cross-attention layer is similar to the self-attention layer of the encoder. The only difference is that the cross-attention layer is used to learn the relationships between the elements of the input sequence and the elements of the output sequence.

```{figure} images/3_transformers/cross-attention-endec.png
---
width: 100%
name: cross-attention-endec
alt: cross-attention-endec
---
Encoder-decoder model showing the cross-attention layer. Image source [illustrated-transformer](http://jalammar.github.io/illustrated-transformer/)
```

{numref}`cross-attention-endec` shows an example of an encoder-decoder model. The input sequence is first converted into a sequence of vector embeddings $X = \{x_1, x_2, ..., x_n\}$ using an embedding layer. The embeddings are then processed by the encoder layers. The output of the encoder is the output embeddings $Y = \{y_1, y_2, ..., y_n\}$ of the last layer. The output embeddings are then processed by the decoder layers. Here we have an **additional** attention layer that leverage **key** and **value** vectors from the encoder to combine the output embeddings of the encoder with the output embeddings of the decoder. The output of the decoder is the output embeddings $Z = \{z_1, z_2, ..., z_n\}$ of the last layer.
The example reported in {numref}`cross-attention-endec` is in the context of NLP. However, the very same architecture can be used in the context of audio, images, or multi-modal data (e.g., an audio sequence is encoded into a sequence of embeddings and then decoded into a sequence of text for speech recognition) {cite:ps}`radford2023robust`.

## An Encoder-Decoder Model in PyTorch

We have seen how to design the different components of the transformer architecture. In this section, we will see how to implement an encoder-decoder model in pure PyTorch.

### Embedding Layer

The embedding layer is a simple linear layer that maps each element of the input sequence to a vector of a specific size. The size of the vector is called the *embedding size* and is (for convenience) a power of 2, or at least divisible by 2 and 3.

```{code-block} python
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        return self.embedding(x)
```

When calling the embedding layer, we pass the input sequence in terms of **ids**. The ids are integers that represent the elements of the sequence. For example, in the case of text, the ids are the indices of the words in the vocabulary. In the case of audio, the ids are the indices of the frames in the vocabulary. Depending on the domain, we may not need an embedding layer. For example, in the case of images, we can use the pixel values as input to the model.

Let's take a look at the input-output shape of the embedding layer.
- **Input**. The input of the embedding layer is a sequence of ids. The shape of the input is $(B, S)$ where $B$ is the batch size and $S$ is the sequence length.
- **Output**. The output of the embedding layer is a sequence of vector embeddings. The shape of the output is $(B, S, E)$ where $B$ is the batch size, $S$ is the sequence length, and $E$ is the embedding size.

### Positional Encoding

The positional encoding is a technique that consists of **injecting information about the position** of each element of the sequence into the model. There are different implementations of positional encoding. The traditional implementation is based on sinusoidal functions. For each position $i$ of the input sequence, we compute a vector $PE_i$ of the same size as the embeddings. The vector $PE_i$ is then added to the embeddings $x_i$ to produce the final embeddings $x_i + PE_i$.

```{code-block} python
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```

The positional encoding is implemented as a PyTorch module. It is initialized with the embedding size and the dropout rate. The positional encoding is computed in the `forward` method. We define a set of frequencies $f$ and compute the vector $PE_i$ accordingly (review the Section on {ref}`pos_encoding_section` for more details).

When passing through the positional encoding layer, the input embeddings are summed up with the positional encoding vectors, therefore their shape does not change $$(B, S, E) \rightarrow (B, S, E)$$.

There are several other implementations of positional encoding. For example, we can have a learnable positional encoding that is learned during training. It is implemented using an embedding layer.

```{code-block} python
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, embedding_size)

    def forward(self, x):
        position = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        position = position.unsqueeze(0).expand_as(x)
        x = x + self.embedding(position)
        return self.dropout(x)
```

In this case, the positional encoding is learned during training. We should pay attention to the fact that this implementation does not force the model to learn the time-related patterns in the data. The name *positional encoding* is only used for convenience as it may learn **other** patterns in the data.

### Attention Mechanism

Diving inside the transformer architecture, we need to implement the attention mechanism. The attention mechanism is a mechanism that allows the model to learn the relationships between the different elements of the sequence. The process can be divided into three steps:
1. **Query, Key, and Value**. The input embeddings are first *split* into three vectors: the query vector, the key vector, and the value vector.
2. **Attention**. The query vector is compared to the key vector to produce a score, e.g., a float value between 0 and 1. The score is then used to compute a weighted average of the value vector. The weighted average is called the *attention vector*.
3. **Output**. The attention vector is then processed by a linear layer to produce the output vector.

```{code-block} python
import torch
import torch.nn as nn

class MHSA(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        self.scale = self.head_size ** -0.5

        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.out = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        B, S, E = x.size()
        Q = self.query(x).view(B, S, self.num_heads, self.head_size).transpose(1, 2)
        K = self.key(x).view(B, S, self.num_heads, self.head_size).transpose(1, 2)
        V = self.value(x).view(B, S, self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.softmax(dim=-1)
        scores = scores.matmul(V).transpose(1, 2).reshape(B, S, E)
        return self.out(scores)
```

The class `MHSA` implements the multi-head attention mechanism. The input of the attention mechanism is a sequence of embeddings. The embeddings are first split into three vectors: the query vector, the key vector, and the value vector. The query, key, and value vectors are then processed by three linear layers. The output of the linear layers is then split into multiple heads. The number of heads is a **hyperparameter** of the model. The output of the attention mechanism is the concatenation of the output of each head. The output of the attention mechanism is then processed by a linear layer to produce the output embeddings.

Similarly to the positional encoding, the MHSA is implemented as a PyTorch module. The input and output of the MHSA are embeddings, therefore their shape does not change $$(B, S, E) \rightarrow (B, S, E)$$.

````{admonition} Test MHSA implementation
:class: tip
We can test the implementation of the MHSA module by creating a random input tensor and passing it through the module.

```{code-block} python
import torch
from mhsa import MHSA # import the MHSA module

mhsa = MHSA(embedding_size=512, num_heads=8) # create the MHSA module
x = torch.randn(2, 10, 512) # create a random input tensor
y = mhsa(x) # pass the input tensor through the MHSA module
print(y.shape) # print the shape of the output tensor
```
````

### Cross-Attention

The cross-attention layer is similar to the self-attention layer of the encoder. The only difference is that the cross-attention layer is used to learn the relationships between the elements of the input sequence and the elements of the output sequence.

```{code-block} python
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        self.scale = self.head_size ** -0.5

        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.out = nn.Linear(embedding_size, embedding_size)

    def forward(self, x, y):
        '''
        x: input embeddings
        y: vector to "cross-attend" to
        '''
        B, S, E = x.size()
        Q = self.query(x).view(B, S, self.num_heads, self.head_size).transpose(1, 2) # queries are computed from x
        K = self.key(y).view(B, S, self.num_heads, self.head_size).transpose(1, 2) # keys are computed from y
        V = self.value(y).view(B, S, self.num_heads, self.head_size).transpose(1, 2) # values are computed from y

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.softmax(dim=-1)
        scores = scores.matmul(V).transpose(1, 2).reshape(B, S, E)
        return self.out(scores)
```

As noted also in code comments, the queries are computed from the input embeddings $x$ while the keys and values are computed from the vector $y$. All the other considerations are the same as the self-attention layer (e.g., the output of the cross-attention layer is the concatenation of the output of each head).

### Feed-Forward and Residual Connections

After the attention mechanism, the output embeddings are processed by a feed-forward network. The feed-forward network is a simple linear layer followed by a non-linear activation function (e.g., ReLU).

Similarly to what we have seen with ResNets {cite:ps}`he2015deep`, in each layer of the encoder and decoder, there are *residual connections* that sum up the output of a sub-layer with the input of the sub-layer.

```{code-block} python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Residual(nn.Module):
    def __init__(self, sublayer, dropout):
        super().__init__()
        self.sublayer = sublayer
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(sublayer.size)

    def forward(self, x):
        return x + self.dropout(self.sublayer(self.norm(x)))
```

The class `FeedForward` implements the feed-forward network. The class `Residual` implements the residual connections. The Residual class is usually not used in real implementations of transformer models.
Instead, the residual connections are implemented directly in the encoder and decoder layers.

```{code-block} python

# ... other code ...
x # is the tensor we want to add for the residual connection
x_ff = self.ff(x) # pass x through the feed-forward network
x_ff = self.dropout(x_ff) # apply dropout
x = x + x_ff # add the residual connection
```

### Encoder and Decoder Models

At this point we have all the ingredients to create an encoder transformer layer. The encoder model is composed of:
- **Embedding layer**. The embedding layer converts the input sequence into a sequence of vector embeddings. If we deal with vectorized data (e.g., images), we can skip this step or implement it differently {cite:ps}`liu2020mockingjay`.
- **Positional encoding**. The positional encoding injects information about the position of each element of the sequence into the model.
- **Encoder layers**. The encoder layers process the embeddings using:
    - **Multi-head attention**. The multi-head attention mechanism is used to learn the relationships between the different elements of the sequence.
    - **Feed-forward network**. The feed-forward network is used to process the output of the multi-head attention mechanism.
    - **Residual connections**. The residual connections are used to sum up the output of the multi-head attention mechanism with the input embeddings.
    - **Layer normalization**. The layer normalization is used to normalize the output of the encoder layer.
- **Output**. The output of the encoder is the output embeddings of the last encoder layer.

A stack of encoder layers is used to create the encoder model. The output of the encoder is the output embeddings of the last encoder layer.

```{code-block} python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout):
        super().__init__()
        self.mhsa = MHSA(embedding_size, num_heads)
        self.ff = FeedForward(embedding_size, hidden_size)
        self.residual1 = Residual(self.mhsa, dropout)
        self.residual2 = Residual(self.ff, dropout)

    def forward(self, x):
        x = self.residual1(x)
        x = self.residual2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embedding_size, num_heads, hidden_size, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

The class `EncoderLayer` implements a single encoder layer. The class `Encoder` implements a stack of encoder layers. The output of the encoder is the output embeddings of the last encoder layer.

```{admonition} Note on the number of layers
:class: tip
The number of layers is a **hyperparameter** of the model. The number of layers is usually between 6 and 12. The number of layers is usually the same for the encoder and decoder. However, it is possible to use a different number of layers for the encoder and decoder.
```

# Conclusion

In this chapter, we have seen all the components behind one of the most popular deep learning architectures: the transformer (encoder decoder) and its encoder-only and decoder-only variants. We have seen how to implement the different components of the transformer architecture in pure PyTorch.

This architecture has basically revolutionized the field of deep learning. It has been used in many different domains (e.g., NLP, audio, images, multi-modal data, etc.) and has achieved state-of-the-art results in many different tasks (e.g., speech recognition, machine translation, image classification, etc.).

## Vision Transformers (ViT): Transformers for Images

While transformers were originally designed for sequential data like text, the **Vision Transformer (ViT)** {cite:ps}`dosovitskiy2020image` demonstrated that transformers can also work exceptionally well for images by treating images as sequences of patches.

### How ViT Works

**1. Image Patchification**

Instead of processing individual pixels, ViT divides the image into fixed-size patches:

```
Image: 224 x 224 x 3 (RGB)
Patch size: 16 x 16
Number of patches: (224/16) x (224/16) = 14 x 14 = 196 patches
```

Each 16x16x3 patch is flattened into a vector of size 768 (16 × 16 × 3 = 768).

**2. Linear Projection**

Each flattened patch is linearly projected to the embedding dimension (e.g., 768):
```python
self.patch_embedding = nn.Linear(patch_size * patch_size * 3, embedding_dim)
```

**3. Add Position Embeddings**

Since transformers don't inherently understand spatial relationships, we add learnable position embeddings to each patch embedding:
```python
self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
```

**4. Add Class Token**

A special learnable [CLS] token is prepended to the sequence. After processing through transformer layers, this token's representation is used for classification:
```python
self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
```

**5. Transformer Encoder**

The sequence of patch embeddings (including the [CLS] token) is processed by standard transformer encoder layers.

**6. Classification Head**

The output corresponding to the [CLS] token is passed through a classification head:
```python
self.mlp_head = nn.Linear(embedding_dim, num_classes)
```

### ViT Architecture Summary

```{code-block} python
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 embedding_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_dim, embedding_dim)
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers
        )
        
        # Classification head
        self.mlp_head = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch, 3, 224, 224)
        batch_size = x.shape[0]
        
        # Split into patches and flatten
        # x shape: (batch, num_patches, patch_dim)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
        
        # Linear projection
        x = self.patch_embedding(x)  # (batch, num_patches, embedding_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, embedding_dim)
        
        # Add position embeddings
        x = x + self.position_embedding
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Classification (use CLS token)
        cls_output = x[:, 0]  # (batch, embedding_dim)
        return self.mlp_head(cls_output)  # (batch, num_classes)
```

### ViT vs CNNs

**Advantages of ViT:**
- **Global receptive field**: Every patch can attend to every other patch from the first layer
- **Scalability**: Performance improves consistently with more data and larger models
- **Flexibility**: Same architecture works across different vision tasks
- **Interpretability**: Attention weights show which patches the model focuses on

**Disadvantages:**
- **Data hungry**: Requires large datasets (ImageNet-21k or JFT-300M) for good performance
- **Lack of inductive bias**: CNNs have built-in translation equivariance; ViTs must learn this from data
- **Computational cost**: Quadratic complexity in number of patches

**Hybrid approaches:** Models like Swin Transformer combine the best of both worlds, using shifted windows to limit attention computation while maintaining hierarchical feature learning like CNNs.

### When to Use ViT

- **Large datasets available**: ViT excels when you have millions of training images
- **Transfer learning**: Pre-trained ViT models work very well for fine-tuning on smaller datasets
- **Need global context**: Tasks requiring understanding of relationships across the entire image
- **Unified architecture**: Want to use the same model architecture across vision and language tasks

**For smaller datasets:** Consider using CNNs or hybrid architectures, or use heavily pre-trained ViT models with careful fine-tuning.

In the next chapter, we will get our hands dirty and we will see how to use the transformer architecture in practice, both starting from scratch and using pre-trained models.