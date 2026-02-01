#!/bin/bash

# Image Download Script for DL4SV Course Slides
# Replace the URLs with actual image URLs and run this script to download all missing images

SLIDES_DIR="/Users/mlaquatra/courses/PHD-UKE/DL4SV/slides/images"

echo "Starting image download..."

# ============================================
# MODULE 2: CNNs
# ============================================

# CNN Architecture Overview
curl -o "$SLIDES_DIR/2_cnns/cnn_architecture.png" \
  "REPLACE_WITH_URL"

# Convolution Operation (animated)
curl -o "$SLIDES_DIR/2_cnns/convolution.gif" \
  "REPLACE_WITH_URL"

# Convolution with Stride
curl -o "$SLIDES_DIR/2_cnns/stride.png" \
  "REPLACE_WITH_URL"

# Convolution with Padding
curl -o "$SLIDES_DIR/2_cnns/padding.png" \
  "REPLACE_WITH_URL"

# Pooling Operation (animated)
curl -o "$SLIDES_DIR/2_cnns/pooling.gif" \
  "REPLACE_WITH_URL"

# Max Pooling Example
curl -o "$SLIDES_DIR/2_cnns/max_pooling.png" \
  "REPLACE_WITH_URL"

# LeNet Architecture
curl -o "$SLIDES_DIR/2_cnns/lenet.png" \
  "REPLACE_WITH_URL"

# AlexNet Architecture
curl -o "$SLIDES_DIR/2_cnns/alexnet.png" \
  "REPLACE_WITH_URL"

# VGG Architecture
curl -o "$SLIDES_DIR/2_cnns/vgg.png" \
  "REPLACE_WITH_URL"

# ResNet Residual Block
curl -o "$SLIDES_DIR/2_cnns/residual_block.png" \
  "REPLACE_WITH_URL"

# Data Augmentation Examples
curl -o "$SLIDES_DIR/2_cnns/data_augmentation.png" \
  "REPLACE_WITH_URL"

# Audio Spectrogram Visualization
curl -o "$SLIDES_DIR/2_cnns/audio_spectrogram.png" \
  "REPLACE_WITH_URL"

# ============================================
# MODULE 3: Transformers
# ============================================

# Self-Attention Mechanism
curl -o "$SLIDES_DIR/3_transformers/self_attention.png" \
  "REPLACE_WITH_URL"

# Multi-Head Attention
curl -o "$SLIDES_DIR/3_transformers/multihead_attention.png" \
  "REPLACE_WITH_URL"

# Positional Encoding Visualization
curl -o "$SLIDES_DIR/3_transformers/positional_encoding.png" \
  "REPLACE_WITH_URL"

# Transformer Block Diagram
curl -o "$SLIDES_DIR/3_transformers/transformer_block.png" \
  "REPLACE_WITH_URL"

# Complete Transformer Architecture
curl -o "$SLIDES_DIR/3_transformers/transformer_architecture.png" \
  "REPLACE_WITH_URL"

# Vision Transformer (ViT) Architecture
curl -o "$SLIDES_DIR/3_transformers/vit_architecture.png" \
  "REPLACE_WITH_URL"

# Attention Visualization Example
curl -o "$SLIDES_DIR/3_transformers/attention_example.png" \
  "REPLACE_WITH_URL"

# ============================================
# MODULE 4: Libraries
# ============================================

# PyTorch Logo/Icon
curl -o "$SLIDES_DIR/4_libraries/pytorch_logo.png" \
  "REPLACE_WITH_URL"

# Training Loop Diagram
curl -o "$SLIDES_DIR/4_libraries/training_loop.png" \
  "REPLACE_WITH_URL"

# Learning Rate Schedule
curl -o "$SLIDES_DIR/4_libraries/lr_schedule.png" \
  "REPLACE_WITH_URL"

# CometML Dashboard Example
curl -o "$SLIDES_DIR/4_libraries/cometml_dashboard.png" \
  "REPLACE_WITH_URL"

# ============================================
# MODULE 5: Applications
# ============================================

# Object Detection Example
curl -o "$SLIDES_DIR/5_applications/object_detection.png" \
  "REPLACE_WITH_URL"

# Semantic Segmentation Example
curl -o "$SLIDES_DIR/5_applications/segmentation.png" \
  "REPLACE_WITH_URL"

# U-Net Architecture
curl -o "$SLIDES_DIR/5_applications/unet.png" \
  "REPLACE_WITH_URL"

# Speech Recognition Pipeline
curl -o "$SLIDES_DIR/5_applications/asr_pipeline.png" \
  "REPLACE_WITH_URL"

# Deployment Architecture
curl -o "$SLIDES_DIR/5_applications/deployment_architecture.png" \
  "REPLACE_WITH_URL"

# Model Optimization Comparison
curl -o "$SLIDES_DIR/5_applications/model_optimization.png" \
  "REPLACE_WITH_URL"

echo ""
echo "✅ Image download complete!"
echo "Check $SLIDES_DIR for downloaded images"
