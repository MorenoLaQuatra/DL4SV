# Applications and final projects

```{figure} images/5_applications/cover.png
---
width: 50%
name: cover
alt: cover
---
Image generated using [OpenDALL-E](https://huggingface.co/spaces/mrfakename/OpenDalleV1.1-GPU-Demo)
```

Processing images and audio data is at the core of many applications. This chapter will overview some of the most common applications in this field and how they can be implemented using the tools presented in this book.

# Computer vision

Computer vision is the field of computer science that deals with the automatic extraction of information from images. It is a very broad field that includes many different tasks:
- Image classification: assign a label to an image (e.g. which kind of plant is in a picture).
- Object detection: detect specific objects in an image (e.g. cars, pedestrians, etc.).
- Image segmentation: assign a label to each pixel of an image (e.g. which pixels belong to a car).
- Image generation: generate new images (e.g. generate a picture given a text description).
- Image captioning: generate a text description of an image (e.g. describe the content of a picture).
- ... many more!

## Image classification

Image classification is the task of assigning a label to an image. For example, given an image of a dog, the goal is to assign the label "dog" to it. Both CNNs and transformers can be used for image classification. 

**CNN implementation of image classification**

The following code shows how to implement image classification using a CNN. The code is based on the [PyTorch tutorial on image classification](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

```{code-block} python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the data
dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())

trainset, valset = torch.utils.data.random_split(dataset, [40000, 10000])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                            shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transforms.ToTensor()) 
            
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the model
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(net, trainloader, criterion, optimizer):
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    return running_loss / len(trainloader)

def evaluate(net, valloader, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(valloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(valloader), correct / total

# Train the model
net.to(device)

for epoch in range(10):  # loop over the dataset multiple times

    train_loss = train_one_epoch(net, trainloader, criterion, optimizer)
    val_loss, val_acc = evaluate(net, valloader, criterion)
    print(f"Epoch {epoch} - Train loss: {train_loss:.3f} - Val loss: {val_loss:.3f} - Val acc: {val_acc:.3f}")

print('Finished Training')

# Evaluate the model on the test set
test_loss, test_acc = evaluate(net, testloader, criterion)
print(f"Test loss: {test_loss:.3f} - Test acc: {test_acc:.3f}")
```

This code uses a standard CNN architecture for image classification. The model is trained on the CIFAR10 dataset, which contains **10 classes** of images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. The model achieves an accuracy of ~58% on the test set.

## Object Detection

Object detection goes beyond classification by not only identifying what objects are in an image, but also **where** they are located. The output is a set of bounding boxes with associated class labels and confidence scores.

**Key concepts:**

1. **Bounding boxes**: Rectangles defined by (x, y, width, height) or (x1, y1, x2, y2)
2. **Confidence scores**: Probability that an object is present in the box
3. **Non-Maximum Suppression (NMS)**: Post-processing to remove duplicate detections
4. **Intersection over Union (IoU)**: Metric to measure overlap between predicted and ground truth boxes

**Popular architectures:**

- **YOLO (You Only Look Once)**: Single-stage detector, very fast, real-time capable
- **Faster R-CNN**: Two-stage detector, slower but more accurate
- **DETR (Detection Transformer)**: Transformer-based, treats detection as a set prediction problem

**Using a pre-trained object detection model:**

```{code-block} python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and prepare image
image = Image.open('street.jpg').convert('RGB')
image_tensor = ToTensor()(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    predictions = model(image_tensor)

# predictions is a list of dictionaries with keys:
# 'boxes': bounding box coordinates [x1, y1, x2, y2]
# 'labels': class labels
# 'scores': confidence scores

# Visualize detections with confidence > 0.5
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

for box, label, score in zip(predictions[0]['boxes'], 
                              predictions[0]['labels'], 
                              predictions[0]['scores']):
    if score > 0.5:
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{COCO_CLASSES[label]}: {score:.2f}', 
                bbox=dict(facecolor='yellow', alpha=0.5))

plt.axis('off')
plt.show()
```

## Semantic Segmentation

Semantic segmentation assigns a class label to **every pixel** in an image. Unlike object detection which uses bounding boxes, segmentation provides pixel-level precision.

**Applications:**
- Medical imaging (tumor segmentation, organ segmentation)
- Autonomous driving (road, pedestrian, vehicle segmentation)
- Satellite imagery analysis
- Background removal

**Popular architectures:**

- **U-Net**: Encoder-decoder with skip connections, widely used in medical imaging
- **Mask R-CNN**: Extension of Faster R-CNN that adds a segmentation branch
- **DeepLab**: Uses atrous convolution for multi-scale feature extraction
- **Segment Anything Model (SAM)**: Foundation model for segmentation

**U-Net architecture overview:**

```
         Input Image (H x W x 3)
              |
         Encoder Path (Contracting)
         [Conv + Pool] x N
              |
         Bottleneck
              |
         Decoder Path (Expanding)
         [UpConv + Concat + Conv] x N
              |
         Output Segmentation Map (H x W x Classes)
```

The key innovation is the **skip connections** that concatenate encoder features with decoder features, preserving spatial information lost during downsampling.

```{code-block} python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)
```

```{admonition} Exercise - implement a ViT transformer for image classification - 45 min
:class: exercise
Implement a ViT transformer for image classification. You can use the pre-trained ViT model from the [HuggingFace model hub](https://huggingface.co/models?pipeline_tag=image-classification) and fine-tune it on the CIFAR10 dataset.

You can use the model tag `google/vit-base-patch16-224` and the feature extractor tag `google/vit-base-patch16-224` to get the model and the feature extractor respectively.
✋ Remember that *each* model has its own feature extractor. The model documentation is available [here](https://huggingface.co/docs/transformers/model_doc/vit).
```

**Solution**

````{admonition} Solution
:class: dropdown

```{code-block} python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ViTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        # CiFAR images are 32x32, ViT requires 224x224
        self.resize = transforms.Resize((224, 224))

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.resize(image)
        image = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"]
        return {
            "pixel_values": image.squeeze(),
            "labels": label
        }

    def __len__(self):
        return len(self.dataset)

# Load the data
def get_cifar_dataloaders():
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transforms.ToTensor())

    trainset, valset = torch.utils.data.random_split(dataset, [40000, 10000])
    trainset = ViTDataset(trainset)
    valset = ViTDataset(valset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                            shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=8,
                                                shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transforms.ToTensor())
    testset = ViTDataset(testset)
            
    testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                            shuffle=False, num_workers=2)
    return trainloader, valloader, testloader

trainloader, valloader, testloader = get_cifar_dataloaders()

# Define the model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=10, ignore_mismatched_sizes=True)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(model, trainloader, criterion, optimizer):
    running_loss = 0.0
    for i, batch in enumerate(tqdm(trainloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(batch["pixel_values"])
        logits = outputs.logits
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    return running_loss / len(trainloader)

def evaluate(model, valloader, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(valloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            logits = outputs.logits
            loss = criterion(logits, labels)
            running_loss += loss.item()
            predicted = torch.argmax(logits, dim=-1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(valloader), correct / total

# Train the model
for epoch in range(10):  # loop over the dataset multiple times
    train_loss = train_one_epoch(model, trainloader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, valloader, criterion)
    print(f"Epoch {epoch} - Train loss: {train_loss:.3f} - Val loss: {val_loss:.3f} - Val acc: {val_acc:.3f}")

print('Finished Training')

# Evaluate the model on the test set
test_loss, test_acc = evaluate(model, testloader, criterion)
print(f"Test loss: {test_loss:.3f} - Test acc: {test_acc:.3f}")
```
````

# Speech Processing

Speech processing is the field of computer science that deals with the automatic extraction of information from audio signals. It is a very broad field that includes many different tasks:
- Speech recognition: convert speech to text.
- Speaker recognition: identify the speaker from a speech signal.
- Speech synthesis: generate speech from text.
- Speech translation: translate speech from one language to another.
- ... many more!

Audio signals can be analyzed using two different representations: the **time-domain** representation and the **frequency-domain** representation. The time-domain representation is the most intuitive one: it represents the amplitude of the signal as a function of time. The frequency-domain representation is obtained by applying a Fourier transform to the time-domain representation. It represents the amplitude of the signal as a function of frequency. 

**Time-frequency representations** are usually treated as images and can be processed using CNNs or transformers. **Time-domain representations**, on the other hand, are time series and can be processed using RNNs or transformers.

## Keyword spotting

Keyword spotting is the task of detecting specific words in an audio signal. For example, given an audio signal, the goal is to detect the word "yes" in it. Both CNNs and transformers can be used for the task. One practical application of keyword spotting is the detection of wake words in smart speakers. For example, the wake word "Alexa" is used to activate the Amazon Echo smart speaker.

```{admonition} Exercise - implement a transformer for keyword spotting - 45 min
:class: exercise

Using the [superb](https://huggingface.co/superb) dataset, implement a transformer for keyword spotting. You can use the pre-trained transformer from the [HuggingFace model hub](https://huggingface.co/models?pipeline_tag=audio-classification) and fine-tune it on the superb dataset. Alternatively, you can implement your own transformer-based model from scratch using the [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) implementation of encoder layers.
```

**Solution**

````{admonition} Solution
:class: dropdown

```{code-block} python

# load dataset
from datasets import load_dataset
train_dataset = load_dataset("superb", "ks", split="train")
val_dataset = load_dataset("superb", "ks", split="validation")
test_dataset = load_dataset("superb", "ks", split="test")

print(train_dataset[0])
print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of validation examples: {len(val_dataset)}")
print(f"Number of testing examples: {len(test_dataset)}")

# Define the model
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
unique_labels = set([example["label"] for example in train_dataset])
num_labels = len(unique_labels)
model = AutoModelForAudioClassification.from_pretrained("microsoft/wavlm-base-plus", num_labels=num_labels)
print(f"Initalized model with {num_labels} labels")


# implement the dataset class
import torch

class KSDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.max_length_in_seconds = 2

    def __getitem__(self, idx):
        audio_array = self.dataset[idx]["audio"]["array"]
        label = self.dataset[idx]["label"]
        audio = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            max_length=self.max_length_in_seconds * 16000,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        ).input_values
        return {
            "input_values": audio.squeeze(),
            "labels": label
        }

    def __len__(self):
        return len(self.dataset)
    
train_ds = KSDataset(train_dataset)
val_ds = KSDataset(val_dataset)
test_ds = KSDataset(test_dataset)

from torch import nn, optim
from tqdm.notebook import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data loaders
trainloader = torch.utils.data.DataLoader(train_ds, batch_size=16,
                                            shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_ds, batch_size=16,
                                            shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(test_ds, batch_size=16,
                                            shuffle=False, num_workers=2)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
lamda_fn = lambda epoch: 0.95 ** epoch
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lamda_fn)

def train_one_epoch(model, trainloader, criterion, optimizer):
    running_loss = 0.0
    for i, batch in enumerate(tqdm(trainloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(batch["input_values"])
        logits = outputs.logits
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    return running_loss / len(trainloader)

def evaluate(model, valloader, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(valloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_values"])
            labels = batch["labels"]
            logits = outputs.logits
            loss = criterion(logits, labels)
            running_loss += loss.item()
            predicted = torch.argmax(logits, dim=-1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(valloader), correct / total

# Train the model
model.to(device)
best_model = None
best_acc = 0.0
for epoch in range(10):  # loop over the dataset multiple times
    train_loss = train_one_epoch(model, trainloader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, valloader, criterion)
    if val_acc > best_acc:
        best_acc = val_acc
        best_model = model.state_dict()
    print(f"Epoch {epoch} - Train loss: {train_loss:.3f} - Val loss: {val_loss:.3f} - Val acc: {val_acc:.3f}")
    scheduler.step()
    
print('Finished Training')

# load the best model
model.load_state_dict(best_model)

# Evaluate the model on the test set
test_loss, test_acc = evaluate(model, testloader, criterion)
print(f"Test loss: {test_loss:.3f} - Test acc: {test_acc:.3f}")
```
````

# Model Deployment

After training a model, the next step is often to deploy it for inference in production. Deployment requires considerations beyond just model accuracy.

## Inference Optimization

**1. Model Export Formats**

**ONNX (Open Neural Network Exchange)**

ONNX is an open format that allows interoperability between different frameworks:

```{code-block} python
import torch
import torch.onnx

# Export PyTorch model to ONNX
model = MyModel()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                               'output': {0: 'batch_size'}})

# Load and run ONNX model
import onnxruntime as ort

ort_session = ort.InferenceSession("model.onnx")
inputs = {ort_session.get_inputs()[0].name: input_array}
outputs = ort_session.run(None, inputs)
```

**TorchScript**

TorchScript creates optimized models that can run without Python:

```{code-block} python
import torch

model = MyModel()
model.eval()

# Method 1: Tracing (records operations during forward pass)
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# Method 2: Scripting (analyzes code structure)
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Load and use
loaded_model = torch.jit.load("model_traced.pt")
output = loaded_model(input_tensor)
```

**2. Quantization**

Reduce model size and increase inference speed by using lower precision (e.g., INT8 instead of FP32):

```{code-block} python
import torch

# Dynamic Quantization (easiest)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Static Quantization (requires calibration data)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Run calibration data through model
torch.quantization.convert(model, inplace=True)
```

**Benefits:** 2-4x smaller model size, 2-4x faster inference, minimal accuracy loss

**3. Model Pruning**

Remove unnecessary weights to reduce model size:

```{code-block} python
import torch.nn.utils.prune as prune

# Prune 30% of weights in a specific layer
prune.l1_unstructured(model.conv1, name='weight', amount=0.3)

# Make pruning permanent
prune.remove(model.conv1, 'weight')
```

## Deployment Strategies

**1. REST API (Flask/FastAPI)**

Simple web service for model inference:

```{code-block} python
from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

app = FastAPI()
model = torch.load('model.pt')
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess
    tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1).item()
    
    return {"prediction": prediction}
```

**2. Batch Processing**

Process large datasets offline:

```{code-block} python
def batch_inference(model, dataloader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
    
    return predictions
```

**3. Edge Deployment**

For mobile or embedded devices:
- Use quantized models
- Consider TensorFlow Lite or PyTorch Mobile
- Optimize for specific hardware (NPUs, DSPs)

## Performance Considerations

**Latency vs Throughput:**
- **Latency**: Time to process a single request (important for real-time applications)
- **Throughput**: Number of requests processed per second (important for batch processing)

**Tips for reducing latency:**
- Use smaller models or knowledge distillation
- Apply quantization
- Use GPUs for acceleration
- Batch multiple requests when possible

**Tips for increasing throughput:**
- Increase batch size (up to memory limits)
- Use multiple GPUs
- Pipeline processing stages

# Ethical Considerations and Responsible AI

As deep learning models become more powerful and widely deployed, it is crucial to consider their ethical implications.

## Bias and Fairness

**Problem:** Models can perpetuate or amplify biases present in training data.

**Examples:**
- Facial recognition systems with lower accuracy for certain demographics
- Speech recognition systems performing worse for non-native speakers
- Hiring algorithms discriminating based on gender or ethnicity

**Mitigation strategies:**
- Ensure diverse and representative training data
- Evaluate model performance across different demographic groups
- Use fairness-aware training objectives
- Regular audits of deployed models
- Involve diverse stakeholders in model development

## Privacy

**Concerns:**
- Models may memorize and leak sensitive training data
- Facial recognition raises surveillance concerns
- Audio models may capture private conversations

**Best practices:**
- Use differential privacy during training
- Minimize data collection and retention
- Anonymize personal information
- Provide opt-out mechanisms
- Comply with regulations (GDPR, CCPA)

## Transparency and Explainability

**Why it matters:**
- Users have a right to understand decisions affecting them
- Debugging and improving models requires understanding their behavior
- Building trust in AI systems

**Approaches:**
- Provide confidence scores with predictions
- Use attention visualization for transformers
- Apply gradient-based attribution methods (e.g., GradCAM for images)
- Develop model cards documenting model capabilities and limitations

## Environmental Impact

**Concern:** Training large models consumes significant energy.

**Example:** Training GPT-3 emitted approximately 552 tons of CO2.

**Sustainable practices:**
- Use pre-trained models when possible (transfer learning)
- Choose energy-efficient hardware
- Consider model size vs performance trade-offs
- Use carbon-aware computing (train during low-carbon periods)
- Report energy consumption in research papers

## Best Practices Summary

1. **Document limitations**: Clearly state what your model can and cannot do
2. **Test rigorously**: Evaluate across diverse scenarios and edge cases
3. **Monitor in production**: Track model performance and potential issues
4. **Plan for failure**: Have fallback mechanisms when models fail
5. **Involve stakeholders**: Include domain experts and affected communities
6. **Regular audits**: Continuously assess bias, fairness, and safety
7. **Responsible disclosure**: Be transparent about model capabilities and risks

```{admonition} Ethical Framework
:class: important
Before deploying a model, ask:
- Who might be harmed by this system?
- What are the potential negative consequences?
- How will we monitor and address issues?
- Do the benefits outweigh the risks?
- Are we treating all users fairly?
```

# Conclusion

In this chapter, we have seen how to use CNNs and transformers for image and audio processing. We covered:

**Computer Vision Applications:**
- Image classification with CNNs and Vision Transformers
- Object detection for locating objects in images
- Semantic segmentation for pixel-level classification

**Speech Processing Applications:**
- Keyword spotting using transformers
- Processing audio with time-frequency representations

**Practical Deployment:**
- Model optimization techniques (ONNX, TorchScript, quantization)
- Deployment strategies (REST APIs, batch processing, edge devices)
- Performance considerations (latency vs throughput)

**Responsible AI:**
- Addressing bias and ensuring fairness
- Protecting privacy in AI systems
- Building transparent and explainable models
- Considering environmental impact

**Key Takeaways:**

1. **Choose the right architecture**: CNNs for spatial data, Transformers for sequential data, hybrid approaches when needed
2. **Transfer learning is powerful**: Start with pre-trained models and fine-tune for your task
3. **Optimize for deployment**: Consider model size, speed, and accuracy trade-offs
4. **Think beyond accuracy**: Consider fairness, privacy, interpretability, and environmental impact
5. **Monitor continuously**: Track model performance and potential issues in production

```{admonition} Next Steps for Your Research
:class: tip
When applying deep learning to your research:
1. Start simple: Begin with baseline models and standard architectures
2. Leverage pre-trained models: Use transfer learning when possible
3. Focus on data quality: Good data is more important than complex models
4. Experiment systematically: Track all experiments and hyperparameters
5. Validate rigorously: Test on diverse scenarios representative of real-world use
6. Consider deployment early: Think about how the model will be used
7. Document thoroughly: Record decisions, limitations, and lessons learned
```

<!-- 
```{admonition} Final project - 1 week
As a final assignment for the course, you are asked to provide a project that uses the tools presented in this course. You can choose any topic and dataset you like, as long as it is related to image or audio processing. You are free to choose the model you want to use (CNN, transformer, etc.). You can use the code from the previous exercises as a starting point.

Create a new repository on GitHub and upload your code there. Structure your code in a way that is easy to understand and to use. 
Within 1 week from the end of the course, you will be asked to submit a report describing your project. The report should include:
- A description of the dataset and the task.
- A brief overview of the model you used.
- A discussion of the results you obtained.
- A summary of the lessons learned and the challenges encountered.
- If relevant, a discussion of the next steps.

You can use LaTeX to write your report. You can use [Overleaf](https://www.overleaf.com/) to write your report online. You can use [this template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn) to write your report.
Submit your report by sending an email to `moreno.laquatra@unikore.it`. It should include a link to your GitHub repository and the PDF of your report (3 pages max). -->

```{admonition} Final project - 1 week
As a final assignment for the course, you are asked to provide a short report presenting an idea on **how and where** you would use the tools presented in this course in your research. You can provide a brief description of the data you would use, the model you would use and the results you would expect to obtain. 
Even if not directly related to image or audio processing, you are free to choose the model you want to use (CNN, transformer, etc.). If you wish, you can provide a draft implementation of your idea. You can use the code from the previous exercises as a starting point.

Use LaTeX to write your report. You can use [Overleaf](https://www.overleaf.com/) to write your report online. You can use [this template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn) to write your report. Submit your report by sending an email to [moreno.laquatra@unikore.it](mailto:moreno.laquatra@unikore.it).
```