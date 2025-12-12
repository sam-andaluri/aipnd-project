#!/usr/bin/env python3
"""
Train a new network on a dataset and save the model as a checkpoint.

Basic usage: python train.py data_directory
Options:
    --save_dir: Set directory to save checkpoints
    --learning_rate: Set learning rate
    --hidden_units: Set hidden units in classifier
    --epochs: Set number of training epochs
    --gpu: Use GPU for training
"""

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os


def get_input_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset')

    parser.add_argument('data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=1024,
                        help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training')

    return parser.parse_args()


def load_data(data_dir):
    """Load and preprocess the data."""
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
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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


def build_model(hidden_units):
    """Build the model with a custom classifier."""
    # Load pre-trained VGG16 model
    model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier 
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


def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    """Train the model."""
    print(f"Training on {device}...")
    model.to(device)

    print_every = 20
    steps = 0

    for epoch in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in dataloaders['train']:
            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Validation pass
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

                running_loss = 0
                model.train()

    print("Training complete!")
    return model


def save_checkpoint(model, image_datasets, epochs, optimizer, save_dir):
    """Save the model checkpoint."""
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'arch': 'vgg16',
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }

    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    """Main function."""
    # Get command line arguments
    args = get_input_args()

    # Set device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        if args.gpu:
            print("GPU not available, using CPU instead.")

    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {args.data_dir}...")
    image_datasets, dataloaders = load_data(args.data_dir)
    print(f"Training samples: {len(image_datasets['train'])}")
    print(f"Validation samples: {len(image_datasets['valid'])}")
    print(f"Test samples: {len(image_datasets['test'])}")

    # Build model
    print("Building model with vgg16 architecture...")
    model = build_model(args.hidden_units)

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train model
    model = train_model(model, dataloaders, criterion, optimizer, args.epochs, device)

    # Save checkpoint
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_checkpoint(model, image_datasets, args.epochs, optimizer, args.save_dir)


if __name__ == '__main__':
    main()
