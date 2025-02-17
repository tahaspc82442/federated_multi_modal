import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import torch

class Dino:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the DINO model from the PyTorch Hub
        self.dinomodel = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device).half()
        self.dinomodel.eval()  # Set the model to evaluation mode

    def forward(self, x):
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Get the patch embeddings by forwarding through the DINO model
            attentions = self.dinomodel.get_intermediate_layers(x, n=1)  # Extract intermediate layers output
            patch_embeddings = attentions[0][:, 1:, :]  # Ignore class token and take the rest
        return patch_embeddings

if __name__ == '__main__':
    # Example usage:
    # Initialize the Dino model
    dino_model = Dino()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example input tensor (batch_size, channels, height, width)
    x = torch.randn(4, 3, 224, 224).to(device).half()  # Assume the input is a 224x224 image

    # Get patch embeddings
    patch_embeddings = dino_model.forward(x)
    print(patch_embeddings.shape)  # Output: (batch_size, num_patches, embedding_dim)


