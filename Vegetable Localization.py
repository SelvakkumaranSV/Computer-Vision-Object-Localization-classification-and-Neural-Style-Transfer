# -----------------------------------------------------------------------------
# LESSON LEARNED: THE TENSOR PACKAGING BUG
#
# The Mistake:
# Extracting the XML class name as a raw string (e.g., "cucumber") and passing
# it directly to the PyTorch DataLoader caused an AttributeError during loss calculation.
#
# The Correction:
# Implemented a class_to_idx dictionary mapping inside the dataset's __init__
# to convert strings into integer indices (0, 1, 2) and carefully wrapped them
# in torch.tensor(dtype=torch.long).
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# LESSON LEARNED: THE 1:1 BOUNDING BOX SCALING BUG
#
# The Mistake:
# Calculating Scale = 224 / Image_size (which equals exactly 1.0). The images
# were successfully resizing to 224x224, but the target bounding box coordinates
# were still scaled to the original, massive image dimensions.
#
# The Correction:
# Extracted the original PIL image dimensions (orig_width, orig_height) and
# calculated independent x and y multipliers to scale the coordinates correctly.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# LESSON LEARNED: THE ACCURACY OVERWRITE BUG
#
# The Mistake:
# Inside the training loop, using `correct_predictions = (predictions==labels).sum().item()`
# overwrote the accuracy tally every single batch. The printed accuracy only
# reflected the final 32 images of the epoch, not the whole dataset.
#
# The Correction:
# Changed the `=` to `+=` to properly accumulate the correct predictions across
# the entire epoch.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# LESSON LEARNED: THE COMPUTATIONAL GRAPH MEMORY LEAK
#
# The Mistake:
# Accumulating running loss with `running_ce_loss += classifier_loss`. This
# secretly stores the entire PyTorch backpropagation mathematical graph in
# memory for every batch, which will eventually crash the GPU (VRAM OOM).
#
# The Correction:
# Appended `.item()` (e.g., `classifier_loss.item()`) to extract just the raw
# float value, completely protecting the memory over long training runs.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# LESSON LEARNED: GRADIENT DROWNING
#
# The Mistake:
# The MSELoss (calculating raw pixels) was ~18,000, while CrossEntropyLoss
# was ~1.1. Adding them together meant the massive bounding box loss completely
# drowned out the classification loss, sending garbage gradients to the classifier.
#
# The Correction:
# Normalized the coordinates to bring MSE down to tiny decimals (0.02) so both
# losses operated on a similar mathematical scale, allowing simultaneous learning.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# LESSON LEARNED: RAW PIXEL PREDICTION
#
# The Mistake:
# Asking the final Linear layer to guess exact pixel coordinates (e.g., 210).
# Neural networks struggle to output large continuous numbers, causing the MSE
# loss to plateau.
#
# The Correction:
# Divided the bounding box coordinates by the original width/height to turn them
# into percentages [0.0 to 1.0]. Added a nn.Sigmoid() activation layer to the
# regression head to physically force network predictions into that exact range.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# LESSON LEARNED: THE "GOLF PUTT" OSCILLATION
#
# The Mistake:
# The model hit 94% accuracy but then started oscillating wildly. The optimizer's
# learning rate (0.001) was too high for the late stages of training, causing
# it to physically step over the mathematical minimum.
#
# The Correction:
# Introduced PyTorch's optim.lr_scheduler.StepLR to systematically cut the
# learning rate in half as the epochs progressed, smoothing out the descent.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
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

import torch
import torch.nn as nn
import torch.optim as optim
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

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.batch2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if in_channels!=out_channels or stride!=1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=stride,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)

        return self.relu(x + identity)

class Resnet(nn.Module):
    def __init__(self,num_classes=3):
        super(Resnet,self).__init__()
        self.initial_processing = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1), ##32,224,224
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0) ##m,32,112,112
        )

        self.res_block1 = ResBlock(in_channels=32,out_channels=64,stride=1) ##64,112,112
        self.res_block2 = ResBlock(in_channels=64,out_channels=128,stride=2) ##128,56,56
        self.res_block3 = ResBlock(in_channels=128,out_channels=256,stride=2) ##256,28,28

        self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(256,num_classes)
        self.bbox_finder = nn.Sequential(
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,4)
        )
    def forward(self,x):
        x = self.initial_processing(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.globalpool(x)
        x = self.flatten(x)
        class_ = self.classifier(x)
        bbox_ = self.bbox_finder(x)
        return class_,bbox_

model = Resnet().to(device)
#print(model)
total_params = sum(torch.numel(param) for param in model.parameters())
#print(total_params)

loss1 = nn.CrossEntropyLoss()
loss2 = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

epochs = 100
Lambda = 1
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
    if (epoch+1)%4==0:
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



