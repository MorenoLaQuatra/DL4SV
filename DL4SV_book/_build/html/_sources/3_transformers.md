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

The transformer architecture is designed for modeling sequential data, such as text, audio, and video. It is based on the idea of self-attention, which is a mechanism that allows the network to learn the relationships between different elements of a sequence. For example, in the case of an audio sequence, the network can learn the relationships between different frames of the audio signal and leverage the correlations between them to perform a task such as speech recognition.

The original Transformer architecture was introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) {cite:ps}`vaswani2017attention`
by Vaswani et al. in 2017. Since then, many variants of the original architecture have been proposed, and transformers have become the state-of-the-art architecture for many tasks. In this chapter, we will cover all the building blocks of the transformer architecture and show how they can be used for different tasks.

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

## Positional Encoding

After the embedding layer, the input sequence is converted into a sequence of vector embeddings. As we can see later, the attention mechanism, at the core of the transformer architecture, does not take into account the position of each element of the sequence. To inject this information into the model, we use a technique called *positional encoding*.

There are different implementations of positional encoding. The traditional implementation is based on sinusoidal functions. For each position $i$ of the input sequence, we compute a vector $PE_i$ of the same size as the embeddings. The vector $PE_i$ is then added to the embeddings $x_i$ to produce the final embeddings $x_i + PE_i$.

**How can we compute the vector $PE_i$?** The vector $PE_i$ is computed using a combination of sinusoidal functions. We define a set of frequencies $f$ and compute the vector $PE_i$ as follows:

$$PE_i = \begin{bmatrix} sin(f_1 \times i) \\ cos(f_1 \times i) \\ sin(f_2 \times i) \\ cos(f_2 \times i) \\ \vdots \\ sin(f_{d/2} \times i) \\ cos(f_{d/2} \times i) \end{bmatrix}$$

where $d$ is the size of the embeddings. The frequencies $f$ are computed as follows:

$$f_i = \frac{1}{10000^{2i/d}}$$

The frequencies $f$ are computed using a geometric progression. The first frequency is $f_1 = 1/10000^{2 \times 1/d}$, the second frequency is $f_2 = 1/10000^{2 \times 2/d}$, and so on. The frequencies are then used to compute the vector $PE_i$. $d$ is the size of the embeddings. The vector $PE_i$ is then added to the embeddings $x_i$ to produce the vector $x_i + PE_i$ that will be the input of the network.


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