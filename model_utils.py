#!/usr/bin/env python3
"""
Utility functions for the image classifier model.
Contains shared functions used by both train.py and predict.py.
"""

import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import os


# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device(use_gpu=True):
    """Get the appropriate device (CUDA, MPS, or CPU)."""
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("GPU not available, using CPU instead.")
    return torch.device("cpu")


def load_data(data_dir):
    """
    Load and preprocess the image data.

    Args:
        data_dir: Path to the data directory containing train/valid/test subdirs

    Returns:
        image_datasets: Dictionary of ImageFolder datasets
        dataloaders: Dictionary of DataLoader objects
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Create dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    }

    return image_datasets, dataloaders


def build_model(hidden_units=1024):
    """
    Build the model with a custom classifier using VGG16.

    Args:
        hidden_units: Number of hidden units in the second layer (default 1024)

    Returns:
        model: The complete model
    """
    # Load pre-trained VGG16 model
    model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier (matching notebook: 25088 -> 4096 -> hidden_units -> 102)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(4096, hidden_units)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.5)),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array.

    Args:
        image: Path to the image file

    Returns:
        np_image: Processed image as numpy array (CxHxW format)
    """
    # Open the image
    pil_image = Image.open(image)

    # Resize the image where shortest side is 256 pixels, keeping aspect ratio
    width, height = pil_image.size
    if width < height:
        new_width = 256
        new_height = int(height * 256 / width)
    else:
        new_height = 256
        new_width = int(width * 256 / height)

    pil_image = pil_image.resize((new_width, new_height))

    # Center crop to 224x224
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert to numpy array and normalize to 0-1
    np_image = np.array(pil_image) / 255.0

    # Normalize with ImageNet mean and std
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    np_image = (np_image - mean) / std

    # Transpose to put color channel first (PyTorch expects CxHxW)
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def save_checkpoint(model, optimizer, epochs, class_to_idx, save_path):
    """
    Save a model checkpoint.

    Args:
        model: The trained model
        optimizer: The optimizer used for training
        epochs: Number of epochs trained
        class_to_idx: Class to index mapping
        save_path: Path to save the checkpoint
    """
    checkpoint = {
        'arch': 'vgg16',
        'class_to_idx': class_to_idx,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(filepath, device):
    """
    Load a checkpoint and rebuild the model.

    Args:
        filepath: Path to the checkpoint file
        device: Device to load the model onto

    Returns:
        model: The rebuilt model
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Load pre-trained VGG16 model
    model = models.vgg16(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Load the classifier from checkpoint
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model
