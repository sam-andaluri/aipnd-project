#!/usr/bin/env python3
"""
Predict flower name from an image using a trained deep learning model.

Basic usage: python predict.py /path/to/image checkpoint
Options:
    --top_k: Return top K most likely classes
    --category_names: Use a mapping of categories to real names
    --gpu: Use GPU for inference
"""

import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
import json


def get_input_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict flower name from an image')

    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None,
                        help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference')

    return parser.parse_args()


def load_checkpoint(filepath, device):
    """Load a checkpoint and rebuild the model."""
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


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array.
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
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose to put color channel first (PyTorch expects CxHxW)
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(image_path, model, device, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    # Process the image
    img = process_image(image_path)

    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    # Move to device
    model.to(device)
    img_tensor = img_tensor.to(device)

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        logps = model.forward(img_tensor)

    # Calculate probabilities
    ps = torch.exp(logps)

    # Get top K probabilities and indices
    top_p, top_indices = ps.topk(topk, dim=1)

    # Convert to lists
    top_p = top_p.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    # Invert the class_to_idx dictionary
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    # Convert indices to class labels
    top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_p, top_classes


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

    # Load model from checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = load_checkpoint(args.checkpoint, device)

    # Make prediction
    print(f"Predicting class for {args.image_path}...")
    probs, classes = predict(args.image_path, model, device, args.top_k)

    # Load category names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        names = [cat_to_name[cls] for cls in classes]
    else:
        names = classes

    # Print results
    print("\nPrediction Results\n")

    for i, (name, prob) in enumerate(zip(names, probs), 1):
        print(f"{i}. {name}: {prob*100:.2f}%")

    # Return top prediction
    print(f"\nMost likely class: {names[0]} ({probs[0]*100:.2f}%)")


if __name__ == '__main__':
    main()
