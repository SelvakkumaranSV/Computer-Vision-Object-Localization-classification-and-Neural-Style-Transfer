# LESSON LEARNED: THE "SQUARE PEG" ARCHITECTURAL LIMIT
#
# The Mistake:
# The model successfully found the center of objects but drew boxy, square-like
# bounding boxes around extreme aspect ratios (like long cucumbers).
#
# The Correction:
# Diagnosed that nn.Flatten() permanently destroys the 2D spatial geometry
# necessary for predicting independent width and height stretches with a single
# Linear layer. This proved why modern object detection requires architectures
# like YOLO that preserve the 2D grid and utilize anchor boxes.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# LESSON LEARNED: THE "GRADIENT SHOCKWAVE" (TRANSFER LEARNING)
#
# The Mistake:
# When swapping to the pre-trained ResNet-18 backbone, the new custom heads
# (classifier and bbox_finder) were initialized with completely random weights.
# Because the pre-trained backbone wasn't frozen, the massive errors from these
# random heads during Epoch 1 sent destructive gradients backward through the
# network. This "shockwave" scrambled ResNet's perfect, pre-trained spatial
# awareness, causing the test loss to explode initially.
#
# The Correction:
# "Froze" the backbone immediately after downloading the pre-trained weights
# by setting requires_grad = False. This acts as a mathematical shield, blocking
# backpropagation from altering the pre-trained ResNet layers and forcing the
# optimizer to ONLY update the weights in the custom final layers.
#
# Code implementation:
# for param in resnet.parameters():
#     param.requires_grad = False
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import datasets, transforms
import os
import xml.etree.ElementTree as ET
from PIL import Image

device = torch.device("cuda")
#print(device)
Image_size = 224
path = "C:/Users/Selvakkumaran S V/Downloads/Vegetables_Localization/training_images/training_images"

class SingleObjectDataset(Dataset):
    def __init__(self, data_folder, transforms=None):
        self.data_folder = data_folder
        self.transforms = transforms

        # 1. Create a list of ONLY the image files in the folder
        self.image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
        self.class_to_idx = {
            'cucumber': 0,
            'eggplant': 1,
            'mushroom': 2
        }

    def __len__(self):
        # The length of the dataset is just the number of images
        return len(self.image_files)

    def __getitem__(self, idx):
        # 2. Get the specific image file name for this index (e.g., "cucumber_1.jpg")
        img_name = self.image_files[idx]

        # 3. Create the corresponding XML file name by swapping the extension
        xml_name = img_name.replace('.jpg', '.xml')

        # 4. Build the full file paths
        img_path = os.path.join(self.data_folder, img_name)
        xml_path = os.path.join(self.data_folder, xml_name)

        # 5. Load the image
        image = Image.open(img_path).convert("RGB")

        # 6. Parse the XML file for the bounding box
        tree = ET.parse(xml_path)
        root = tree.getroot()

        bndbox = root.find('object').find('bndbox')
        xmin = float(bndbox.find('xmin').text)/227
        ymin = float(bndbox.find('ymin').text)/227
        xmax = float(bndbox.find('xmax').text)/227
        ymax = float(bndbox.find('ymax').text)/227

        Scale = Image_size/227
        # Convert coordinates to a list or tensor
        bbox = torch.tensor([xmin, ymin, xmax, ymax],dtype=torch.float32) * Scale

        # Get the class name if you need it (e.g., "cucumber")
        class_name = root.find('object').find('name').text
        class_idx = self.class_to_idx[class_name]
        label = torch.tensor(class_idx, dtype=torch.long)
        # (Apply transforms to your image and bbox here if needed)

        return image, bbox, label

full_set = SingleObjectDataset(path)

class SafeSubset:
    def __init__(self, dataset, indices,transforms):
        self.dataset = dataset
        self.indices = indices
        self.transforms = transforms

    def __getitem__(self, idx):
        image, bbox, label = self.dataset[self.indices[idx]]
        image = self.transforms(image)
        return image,bbox,label

    def __len__(self):
        return len(self.indices)

train_transforms = transforms.Compose([
    transforms.Resize((Image_size,Image_size)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((Image_size,Image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

total_images = len(full_set)
m_train = int(0.7*total_images)
m_test = total_images - m_train

indices = torch.randperm(total_images).tolist()
train_set = SafeSubset(full_set,transforms=train_transforms,indices=indices[:m_train])
test_set = SafeSubset(full_set,transforms=test_transforms,indices=indices[m_train:])

train_loader = DataLoader(train_set,batch_size=32,shuffle=True)
test_loader = DataLoader(test_set,batch_size=32,shuffle=False)

class TransferResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(TransferResNet, self).__init__()

        # 1. Download the pre-trained ResNet-18 (It already knows how to see shapes and grass!)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for param in resnet.parameters():
            param.requires_grad = False

        # 2. Isolate the backbone by stripping off the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()

        # 3. ResNet-18 outputs exactly 512 features before its final layer.
        # We branch those 512 features into your two custom heads!
        self.classifier = nn.Linear(512, num_classes)
        self.bbox_finder = nn.Sequential(
            nn.Linear(512, 4),
            nn.Sigmoid()  # Keeps coordinates between 0 and 1
        )

    def forward(self, x):
        # Pass image through the genius feature extractor
        x = self.backbone(x)
        x = self.flatten(x)

        # Branch into your two heads
        class_out = self.classifier(x)
        bbox_out = self.bbox_finder(x)

        return class_out, bbox_out

model = TransferResNet().to(device)
#print(model)
total_params = sum(torch.numel(param) for param in model.parameters())
#print(total_params)

loss1 = nn.CrossEntropyLoss()
loss2 = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

epochs = 30
Lambda = 0.01
for epoch in range(epochs):
    model.train()
    running_mse_loss = 0
    running_ce_loss = 0
    correct_predictions = 0
    total = 0
    for images, bboxes, labels in train_loader:
        images,bboxes,labels = images.to(device), bboxes.to(device), labels.to(device)

        optimizer.zero_grad()
        class_outputs, bbox_params = model(images)
        classifier_loss = loss1(class_outputs,labels)
        bbox_loss = loss2(bbox_params,bboxes)
        net_loss = classifier_loss + Lambda * bbox_loss
        net_loss.backward()
        optimizer.step()

        running_ce_loss += classifier_loss.item()
        running_mse_loss += bbox_loss.item()
        _,predictions = torch.max(class_outputs,dim=1)
        correct_predictions += (predictions==labels).sum().item()
        total += len(labels)

    scheduler.step()
    if (epoch+1)%1==0:
        average_ce_loss = running_ce_loss/len(train_loader)
        average_mse_loss = running_mse_loss/len(train_loader)
        accuracy = correct_predictions/total *100
        print(f"TRAININGSET: [{epoch+1}/{epochs}]: average CrossEntropyLoss:{average_ce_loss}, average MeanSquaredErrorLoss:{average_mse_loss}, ACCURACY:{accuracy}")

        model.eval()
        running_mse_loss = 0
        running_ce_loss = 0
        correct_predictions = 0
        total = 0
        with torch.no_grad():
            for images, bboxes, labels in test_loader:
                images, bboxes, labels = images.to(device), bboxes.to(device), labels.to(device)

                class_outputs, bbox_params = model(images)
                classifier_loss = loss1(class_outputs, labels)
                bbox_loss = loss2(bbox_params, bboxes)
                net_loss = classifier_loss + Lambda * bbox_loss

                running_ce_loss += classifier_loss.item()
                running_mse_loss += bbox_loss.item()
                _, predictions = torch.max(class_outputs, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total += len(labels)

            average_ce_loss = running_ce_loss / len(test_loader)
            average_mse_loss = running_mse_loss / len(test_loader)
            accuracy = correct_predictions / total *100
            print(f"TESTSET: [{epoch+1}/{epochs}]: average CrossEntropyLoss:{average_ce_loss}, average MeanSquaredErrorLoss:{average_mse_loss}, ACCURACY:{accuracy}")
            print("--------------------------------------------------------------------------------------------------------------------------------------------------")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_predictions(model, test_loader, device, num_images=6):
    """
    Draws the Ground Truth (Green) and Predicted (Red) bounding boxes.
    """
    model.eval()

    # ImageNet normalization stats used in your transforms
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Reverse mapping for your classes
    idx_to_class = {0: 'cucumber', 1: 'eggplant', 2: 'mushroom'}

    # Grab one batch of test data
    images, true_bboxes, true_labels = next(iter(test_loader))
    images = images.to(device)

    # Get model predictions
    with torch.no_grad():
        class_outputs, pred_bboxes = model(images)
        _, pred_labels = torch.max(class_outputs, dim=1)

    # Move tensors back to CPU and convert to numpy for Matplotlib
    images = images.cpu().numpy()
    true_bboxes = true_bboxes.cpu().numpy()
    pred_bboxes = pred_bboxes.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()

    # Set up the plot grid (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(num_images, len(images))):
        ax = axes[i]

        # 1. Un-normalize the image colors so they look normal
        img = images[i].transpose((1, 2, 0))  # Change from [C, H, W] to [H, W, C]
        img = std * img + mean
        img = np.clip(img, 0, 1)  # Force pixel values safely between 0 and 1

        ax.imshow(img)

        # 2. Scale the [0,1] coordinates back up to the 224x224 image size
        # Your Image_size variable is 224
        t_xmin, t_ymin, t_xmax, t_ymax = true_bboxes[i] * 224
        p_xmin, p_ymin, p_xmax, p_ymax = pred_bboxes[i] * 224

        # Calculate widths and heights for Matplotlib patches
        t_width, t_height = t_xmax - t_xmin, t_ymax - t_ymin
        p_width, p_height = p_xmax - p_xmin, p_ymax - p_ymin

        # 3. Draw Ground Truth Box (Green)
        true_rect = patches.Rectangle((t_xmin, t_ymin), t_width, t_height,
                                      linewidth=2, edgecolor='g', facecolor='none', label='True Box')
        ax.add_patch(true_rect)

        # 4. Draw Predicted Box (Red)
        pred_rect = patches.Rectangle((p_xmin, p_ymin), p_width, p_height,
                                      linewidth=2, edgecolor='r', facecolor='none', linestyle='--', label='Pred Box')
        ax.add_patch(pred_rect)

        # 5. Add Titles (Green if correct, Red if wrong)
        true_name = idx_to_class[true_labels[i]]
        pred_name = idx_to_class[pred_labels[i]]
        title_color = 'green' if true_labels[i] == pred_labels[i] else 'red'

        ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=title_color, fontweight='bold')
        ax.axis('off')

    # Add a single legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


# Call the function at the end of your script!
visualize_predictions(model, test_loader, device, num_images=6)