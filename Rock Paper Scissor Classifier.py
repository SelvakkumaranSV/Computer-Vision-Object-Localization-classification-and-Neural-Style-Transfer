import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms,datasets

path = "C:/Users/Selvakkumaran S V/Downloads/RockPaperScissor/rps-cv-images"

full_dataset = datasets.ImageFolder(root=path, transform=None)
#print("Class to Index Mapping:", full_dataset.class_to_idx)

total_images = len(full_dataset)
m_train = int(0.7*total_images)
m_test = total_images - m_train
#print(total_images,m_train,m_test)

torch.manual_seed(42)
train_set, test_set = random_split(full_dataset, [m_train,m_test])

train_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#  Define a helper Wrapper Dataset to apply transforms post-split
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Pull the raw image and label from the subset
        image, label = self.subset[idx]

        # Apply the specific transform here
        if self.transform:
            image = self.transform(image)

        return image, label

train_set_transformed = TransformedSubset(train_set,transform=train_transforms)
test_set_transformed = TransformedSubset(test_set,transform=test_transforms)

test_loader = DataLoader(test_set_transformed,batch_size=32,shuffle=True)
train_loader = DataLoader(train_set_transformed,batch_size=32,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.features_filter = nn.Sequential(
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
            nn.Linear(64*32*32,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,3)
        )

    def forward(self,x):
        x = self.features_filter(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
#print(model)
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)

epochs = 5
best_accuracy = 0
for epoch in range(epochs):
    #print(f"------------------EPOCH{epoch+1}---------------")
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

