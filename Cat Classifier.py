import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import h5py
from PIL import Image
import matplotlib.pyplot as plt

test_file_path = "C:/Users/Selvakkumaran S V/Downloads/test_catvsnoncat.h5"
train_file_path = "C:/Users/Selvakkumaran S V/Downloads/train_catvsnoncat.h5/train_catvsnoncat.h5"
h5_test = h5py.File(test_file_path, "r")
h5_train = h5py.File(train_file_path, "r")

#print("Keys in h5 file:", list(h5_test.keys()))
#print("Keys in h5 train file:", list(h5_train.keys()))
#print(np.array(h5_train["train_set_x"][:]).shape)
#print(np.array(h5_train["train_set_y"][:]).shape)

class CatsDataset(Dataset):
    def __init__(self,h5_path,is_train=True,transform=None):
        super(CatsDataset,self).__init__()
        prefix = "train" if is_train else "test"

        h5_file = h5py.File(h5_path,"r")
        self.x = np.array(h5_file[f"{prefix}_set_x"][:]) ##shape is (209,64,64,3)
        self.y = np.array(h5_file[f"{prefix}_set_y"][:])

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        image = self.x[idx]
        label = self.y[idx]

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label,dtype=torch.float32)

        return image,label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Image_size = 64
Batch_size = 32

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_set = CatsDataset(train_file_path,transform=train_transforms)
test_set = CatsDataset(test_file_path,is_train=False,transform=test_transforms)

train_loader = DataLoader(train_set,batch_size=Batch_size,shuffle=True)
test_loader = DataLoader(test_set,batch_size=Batch_size,shuffle=False)

with h5py.File(test_file_path, "r") as f:
    # 2. Load images and labels into NumPy arrays
    images = np.array(f["test_set_x"][:])  # Change to "train_set_x" if using the train file
    labels = np.array(f["test_set_y"][:])  # Change to "train_set_y" if using the train file
    classes = np.array(f["list_classes"][:])  # Category names (e.g., b'non-cat', b'cat')

#print(f"Total images available: {len(images)}")

# 3. Plot the first few images
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    img = images[i]
    label = labels[i]

    # Decode the class label (e.g., b'cat' -> 'cat')
    class_name = classes[label].decode("utf-8")

    axes[i].imshow(img)
    axes[i].set_title(f"Label: {label} ({class_name})")
    axes[i].axis("off")

#plt.tight_layout()
#plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.features_extract = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
              )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16,128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128,1)
        )

    def forward(self,x):
        x = self.features_extract(x)
        x = self.classifier(x)
        return x

model = CNN().to(device)
#print(model)

total_params = sum(param.numel() for param in model.parameters())
#print("Total no. of parameters for this model is ", total_params)

loss_fun = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

epochs = 50
early_stop = False
best_acc = 0
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct_predictions = 0
    total = 0
    for batch_idx,(images,labels) in enumerate(train_loader):
        (images, labels) = (images.to(device),labels.to(device))

        outputs = model(images).squeeze(1)
        loss = loss_fun(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()

        running_loss += loss.item()
        correct_predictions += (preds==labels).sum().item()
        total += len(labels)

    if (epoch+1)%10 == 0:
        avg_loss = running_loss / len(train_loader)
        accuracy = correct_predictions/total * 100
        print(f"epoch[{epoch+1}/{epochs}]: TRAINING SET ACCURACY: {accuracy}")

        model.eval()
        running_loss = 0
        correct_predictions = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(test_loader):
            (images, labels) = (images.to(device), labels.to(device))

            outputs = model(images).squeeze(1)
            loss = loss_fun(outputs,labels)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            running_loss += loss.item()
            correct_predictions += (preds == labels).sum().item()
            total += len(labels)

        avg_loss = running_loss / len(test_loader)
        accuracy = correct_predictions / total * 100
        #scheduler.step(accuracy)
        if accuracy > best_acc:
            best_acc = accuracy
            #torch.save(model.state_dict(), "best_cat_model.pth")
            #print(f"--> Saved New Best Model with Test Accuracy: {accuracy:.2f}%")
        print(f"epoch[{epoch+1}/{epochs}]: TEST SET ACCURACY: {accuracy}")
        if accuracy >= 95:
            early_stop=True
    if early_stop:
        print("----------------EARLY STOP---------------------")
        break

# 1. Ensure model is in evaluation mode
#loaded_model = CNN().to(device)
#loaded_model.eval()

model.eval()

# 2. Load the saved weights into the model
#loaded_model.load_state_dict(torch.load("best_cat_model.pth"))

# 2. Grab a batch of images and labels from the test loader
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Take only the first 10 images
images_to_plot = images[:10].to(device)
true_labels = labels[:10].to(device)

# 3. Get predictions from the model
with torch.no_grad():
    outputs = model(images_to_plot).squeeze(1)
    probs = torch.sigmoid(outputs)
    preds = (probs >= 0.5).float()

# Move tensors back to CPU and convert to NumPy for plotting
images_to_plot = images_to_plot.cpu()
true_labels = true_labels.cpu()
preds = preds.cpu()
probs = probs.cpu()

# Class names mapping
class_names = {0: "non-cat", 1: "cat"}

# 4. Plot the 10 images in a 2x5 grid
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
axes = axes.flatten()

for i in range(10):
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
    confidence = probs[i].item() if pred_class == 1 else (1 - probs[i].item())

    pred_name = class_names[pred_class]
    true_name = class_names[true_class]

    # Color title Green if correct, Red if wrong
    title_color = "green" if pred_class == true_class else "red"

    axes[i].imshow(img)
    axes[i].set_title(
        f"Pred: {pred_name} ({confidence * 100:.1f}%)\nTrue: {true_name}",
        color=title_color,
        fontsize=10
    )
    axes[i].axis("off")

plt.tight_layout()
plt.show()