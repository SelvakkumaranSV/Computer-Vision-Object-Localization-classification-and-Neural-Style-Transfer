import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader,Dataset, random_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

path = "C:/Users/Selvakkumaran S V/Downloads/Garbage_Classification/garbage_classification"
full_dataset = datasets.ImageFolder(path,transform=None)
#print("Class to index mapping", full_dataset.class_to_idx)

class Subset(Dataset):
    def __init__(self,subset,transform=None):
        super(Subset,self).__init__()
        self.subset = subset    ##the subset is already returned as image,label , but without transgforms
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self,idx):
        image, label = self.subset[idx]
        image = self.transform(image)
        return image, label

total_images = len(full_dataset)
m_train = int(0.7*total_images)
m_test = total_images - m_train

train_raw , test_raw = random_split(full_dataset,[m_train,m_test])

image_size = 128
train_transforms = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

train_set = Subset(train_raw,transform=train_transforms)
test_set = Subset(test_raw,transform=test_transforms)

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
    def __init__(self,num_classes=12):
        super(Resnet,self).__init__()
        self.initial_processing = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0) ##m,32,64,64
        )

        self.res_block1 = ResBlock(in_channels=32,out_channels=64,stride=1) ##64,64,64
        self.res_block2 = ResBlock(in_channels=64,out_channels=128,stride=2) ##128,32,32
        self.res_block3 = ResBlock(in_channels=128,out_channels=256,stride=2) ##256,16,16

        self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(256,num_classes)

    def forward(self,x):
        x = self.initial_processing(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x

model = Resnet().to(device)
#print(model)

total_params = sum(torch.numel(param) for param in model.parameters())
#print(total_params)

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)

epochs = 50
best_accuracy = 0
for epoch in range(epochs):
    print(f"------------------EPOCH{epoch+1}---------------")
    model.train()
    running_loss = 0
    total = 0
    correct_predictions = 0
    for images, labels in train_loader:
        (images,labels) = (images.to(device),labels.to(device))

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fun(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += len(labels)
        _,predictions = torch.max(outputs,dim=1)
        correct_predictions += (predictions==labels).sum().item()

    if (epoch+1) % 1 == 0:
        avg_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total * 100
        print(f"epoch[{epoch + 1}/{epochs}]: TRAINING SET ACCURACY: {accuracy}")

        model.eval()
        running_loss = 0
        correct_predictions = 0
        total = 0
        for images, labels in test_loader:
            (images, labels) = (images.to(device), labels.to(device))

            outputs = model(images)
            loss = loss_fun(outputs, labels)

            running_loss += loss.item()
            total += len(labels)
            _,predictions = torch.max(outputs,dim=1)
            correct_predictions += (predictions==labels).sum().item()

        avg_loss = running_loss / len(test_loader)
        accuracy = correct_predictions / total * 100
        print(f"epoch[{epoch + 1}/{epochs}]: TEST SET ACCURACY: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            #torch.save(model.state_dict(), "best_cat_model.pth")
            #print(f"--> Saved New Best Model with Test Accuracy: {accuracy:.2f}%")
        print("------------------------------------------")

import matplotlib.pyplot as plt
import numpy as np

# 1. Put the model in evaluation mode
model.eval()

# 2. Get a batch of images and labels from the test loader
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Take only the first 20 images
images_to_plot = images[:20].to(device)
true_labels = labels[:20].to(device)

# 3. Get predictions from the model
with torch.no_grad():
    outputs = model(images_to_plot)
    _, preds = torch.max(outputs, dim=1)

# Move tensors back to CPU and convert to NumPy for plotting
images_to_plot = images_to_plot.cpu()
true_labels = true_labels.cpu()
preds = preds.cpu()

# Get the class index-to-name mapping from your full dataset
idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}

# 4. Plot the 20 images in a 4x5 grid
fig, axes = plt.subplots(4, 5, figsize=(16, 12))
axes = axes.flatten()

for i in range(20):
    # Un-normalize the image for clear visualization: (img * std) + mean
    img = images_to_plot[i].clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean

    # Change tensor shape from (C, H, W) to (H, W, C) for matplotlib
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)  # Ensure pixel values stay in valid [0, 1] range

    pred_class = int(preds[i].item())
    true_class = int(true_labels[i].item())

    pred_name = idx_to_class[pred_class]
    true_name = idx_to_class[true_class]

    # Color title Green if correct, Red if wrong
    title_color = "green" if pred_class == true_class else "red"

    axes[i].imshow(img)
    axes[i].set_title(
        f"Pred: {pred_name}\nTrue: {true_name}",
        color=title_color,
        fontsize=10
    )
    axes[i].axis("off")

plt.tight_layout()
plt.show()
