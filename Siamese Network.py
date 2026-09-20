import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. The Shared-Weight Twin Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Convolutional backbone to extract feature embeddings
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        )

        # Fully connected layers to output the final embedding vector
        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),  # Assuming 28x28 input (e.g., MNIST/Omniglot)
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256)  # 256-dimensional embedding
        )

    def forward_once(self, x):
        # Forward pass for a single image
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # Pass both images through the EXACT SAME network (shared weights)
        embedding1 = self.forward_once(input1)
        embedding2 = self.forward_once(input2)
        return embedding1, embedding2


# 2. Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        # Label = 1 if images are the same class, 0 if different
        euclidean_distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)

        # Loss: Push similar together, pull dissimilar apart (up to the margin)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                              2))
        return loss_contrastive