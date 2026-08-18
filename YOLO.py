import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import xml.etree.ElementTree as ET
from PIL import Image

device = torch.device("cuda")
Image_size = 448
path = "C:/Users/Selvakkumaran S V/Downloads/Vegetables_Localization/training_images/training_images"

class SingleObjectDataset(Dataset):
    def __init__(self, data_folder, transforms=None):
        self.data_folder = data_folder
        self.transforms = transforms

        self.image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
        self.class_to_idx = {
            'cucumber': 0,
            'eggplant': 1,
            'mushroom': 2
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        xml_name = img_name.replace('.jpg', '.xml')

        img_path = os.path.join(self.data_folder, img_name)
        xml_path = os.path.join(self.data_folder, xml_name)

        image = Image.open(img_path).convert("RGB")
        original_width, original_height = image.size

        tree = ET.parse(xml_path)
        root = tree.getroot()

        bndbox = root.find('object').find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Dynamic scaling based on true image dimensions
        scale_x = Image_size / original_width    ##in terms of whole image ratio
        scale_y = Image_size / original_height   ##in terms of whole image ratio

        xmin = xmin * scale_x
        xmax = xmax * scale_x
        ymin = ymin * scale_y
        ymax = ymax * scale_y

        bx = (xmax + xmin) / 2
        by = (ymax + ymin) / 2
        bh = (ymax - ymin)
        bw = (xmax - xmin)

        bbox = torch.tensor([bx, by, bh, bw], dtype=torch.float32)

        class_name = root.find('object').find('name').text
        class_idx = self.class_to_idx[class_name]
        label = torch.tensor(class_idx, dtype=torch.long)

        return image, bbox, label

full_set = SingleObjectDataset(path)

class SafeSubset:
    def __init__(self, dataset, indices, transforms):
        self.dataset = dataset
        self.indices = indices
        self.transforms = transforms

    def __getitem__(self, idx):
        image, bbox, label = self.dataset[self.indices[idx]]
        image = self.transforms(image)
        return image, bbox, label

    def __len__(self):
        return len(self.indices)

train_transforms = transforms.Compose([
    transforms.Resize((Image_size, Image_size)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((Image_size, Image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

total_images = len(full_set)
m_train = int(0.7 * total_images)
m_test = total_images - m_train

indices = torch.randperm(total_images).tolist()
train_set = SafeSubset(full_set, transforms=train_transforms, indices=indices[:m_train])
test_set = SafeSubset(full_set, transforms=test_transforms, indices=indices[m_train:])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

class CONVBLOCK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CONVBLOCK, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv_block(x)

class MiniDarkNet(nn.Module):
    def __init__(self, num_classes, num_boxes):
        super(MiniDarkNet, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.layer1 = CONVBLOCK(in_channels=3, out_channels=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = CONVBLOCK(in_channels=16, out_channels=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = CONVBLOCK(in_channels=32, out_channels=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4 = CONVBLOCK(in_channels=64, out_channels=128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer5 = CONVBLOCK(in_channels=128, out_channels=256)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer6 = CONVBLOCK(in_channels=256, out_channels=512)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer7 = nn.Conv2d(in_channels=512, out_channels=(num_classes + 5 * num_boxes), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer4(x)
        x = self.pool4(x)
        x = self.layer5(x)
        x = self.pool5(x)
        x = self.layer6(x)
        x = self.pool6(x)
        x = self.layer7(x)
        return x

class YOLO_LOSS(nn.Module):
    def __init__(self, image_size=448):
        super(YOLO_LOSS, self).__init__()
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.Image_size = image_size

    def IoU(self, bx, by, bh, bw, box_pred):
        # Linear IoU for accurate box selection
        x_max = bx + bw * 7 / 2
        x_min = bx - bw * 7 / 2
        y_max = by + bh * 7 / 2
        y_min = by - bh * 7 / 2

        Bx = box_pred[..., 1]
        By = box_pred[..., 2]
        Bh = box_pred[..., 3]
        Bw = box_pred[..., 4]

        X_max = Bx + Bw * 7 / 2
        X_min = Bx - Bw * 7 / 2
        Y_max = By + Bh * 7 / 2
        Y_min = By - Bh * 7 / 2

        Area1 = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)
        Area2 = (X_max - X_min).clamp(min=0) * (Y_max - Y_min).clamp(min=0)

        inter_x_min = torch.max(x_min, X_min)
        inter_y_min = torch.max(y_min, Y_min)
        inter_x_max = torch.min(x_max, X_max)
        inter_y_max = torch.min(y_max, Y_max)

        inter_width = (inter_x_max - inter_x_min).clamp(min=0)
        inter_height = (inter_y_max - inter_y_min).clamp(min=0)

        intersection_area = inter_width * inter_height
        union_area = Area1 + Area2 - intersection_area

        iou = intersection_area / (union_area + 1e-6)
        return iou

    def forward(self, output, bbox, label, epoch_no):
        output = output.permute(0, 2, 3, 1)
        batch_size = output.shape[0]
        batch_idx = torch.arange(batch_size, device=output.device)

        box1_pred_raw = output[..., 0:5]
        box2_pred_raw = output[..., 5:10]
        class_pred_raw = output[..., 10:]

        box1_pred = torch.sigmoid(box1_pred_raw)
        box2_pred = torch.sigmoid(box2_pred_raw)

        bx = bbox[:, 0]
        by = bbox[:, 1]
        bh = bbox[:, 2]
        bw = bbox[:, 3]

        x_cell = (bx * 7 / self.Image_size).long()
        y_cell = (by * 7 / self.Image_size).long()
        bx = (bx * 7) / self.Image_size - x_cell
        by = (by * 7) / self.Image_size - y_cell
        bw = bw / self.Image_size
        bh = bh / self.Image_size

        box1_responsible = box1_pred[batch_idx, y_cell, x_cell, :]
        box2_responsible = box2_pred[batch_idx, y_cell, x_cell, :]

        IoU1 = self.IoU(bx, by, bh, bw, box1_responsible)
        IoU2 = self.IoU(bx, by, bh, bw, box2_responsible)

        box1_won = (IoU1 > IoU2).float().unsqueeze(1)
        box2_won = (IoU2 >= IoU1).float().unsqueeze(1)

        winner_box = (box1_responsible * box1_won) + (box2_responsible * box2_won)

        loss_fun = nn.MSELoss(reduction='sum')

        target_center = torch.stack([bx, by], dim=1)
        target_shape = torch.stack([bh, bw], dim=1)

        center_loss = loss_fun(winner_box[:, 1:3], target_center)
        shape_loss = loss_fun(
            torch.sqrt(torch.abs(winner_box[:, 3:5]) + 1e-6),
            torch.sqrt(target_shape + 1e-6)
        )
        object_loss = loss_fun(winner_box[:, 0], torch.ones_like(winner_box[:, 0]))

        obj_mask = torch.zeros((batch_size, 7, 7), device=output.device)
        obj_mask[batch_idx, y_cell, x_cell] = 1.0
        noobj_mask = 1.0 - obj_mask

        noobj_mask_box1 = noobj_mask + (obj_mask * box2_won.view(batch_size, 1, 1))
        noobj_mask_box2 = noobj_mask + (obj_mask * box1_won.view(batch_size, 1, 1))

        no_object_loss = (
                ((box1_pred[..., 0] * noobj_mask_box1) ** 2).mean()
                +
                ((box2_pred[..., 0] * noobj_mask_box2) ** 2).mean()
        )

        # Classification Loss using CrossEntropy with raw logits
        ce_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        class_pred_responsible = class_pred_raw[batch_idx, y_cell, x_cell, :]
        class_loss = ce_loss_fn(class_pred_responsible, label)

        total_loss = self.lambda_coord * (center_loss + shape_loss) + \
                     self.lambda_noobj * no_object_loss + \
                     object_loss + \
                     class_loss

        if (epoch_no+1)%20==0:
            print(f"Coord_loss:{center_loss + shape_loss}, Object_loss:{object_loss}, No_Object_loss:{no_object_loss}, Class_loss:{class_loss}")

        return total_loss

model = MiniDarkNet(3, 2).to(device)
loss_fun = YOLO_LOSS(image_size=448)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

epochs = 500

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, bboxes, labels in train_loader:
        images, bboxes, labels = images.to(device), bboxes.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fun(outputs, bboxes, labels, epoch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    scheduler.step()
    print(f"[{epoch+1}/{epochs}]: {running_loss / len(train_loader)}")

torch.save(model.state_dict(), "yolo_vegetables_final.pth")
print("Model saved to yolo_vegetables_final.pth")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

model.eval()

def calculate_iou(box1, box2):
    # [x_center, y_center, width, height]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, x1_max = x1 - w1/2, x1 + w1/2
    y1_min, y1_max = y1 - h1/2, y1 + h1/2
    x2_min, x2_max = x2 - w2/2, x2 + w2/2
    y2_min, y2_max = y2 - h2/2, y2 + h2/2

    inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h

    union = w1*h1 + w2*h2 - inter
    return inter / (union + 1e-6)


# Get batch
images, bboxes, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)

outputs = outputs.permute(0, 2, 3, 1)

idx_to_class = {
    0: "cucumber",
    1: "eggplant",
    2: "mushroom"
}

mean = torch.tensor([0.485, 0.456, 0.406],
                    device=device).view(3,1,1)
std = torch.tensor([0.229, 0.224, 0.225],
                   device=device).view(3,1,1)

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for i in range(min(10, len(images))):

    pred = outputs[i]

    # Find best of 7x7x2
    confs = torch.stack([
        torch.sigmoid(pred[..., 0]),
        torch.sigmoid(pred[..., 5])
    ], dim=-1)

    max_idx = torch.argmax(confs)

    y = (max_idx // 14).item()
    x = ((max_idx // 2) % 7).item()
    box_idx = (max_idx % 2).item()

    # Extract selected box
    if box_idx == 0:
        raw_box = pred[y, x, 1:5]
    else:
        raw_box = pred[y, x, 6:10]

    # Same activation as training
    bx = torch.sigmoid(raw_box[0]).item()
    by = torch.sigmoid(raw_box[1]).item()
    bh = torch.sigmoid(raw_box[2]).item()
    bw = torch.sigmoid(raw_box[3]).item()

    confidence = confs[y, x, box_idx].item()

    # Local -> global
    cell = Image_size / 7

    pred_x = (x + bx) * cell
    pred_y = (y + by) * cell
    pred_w = bw * Image_size
    pred_h = bh * Image_size

    # Ground truth: [x, y, h, w]
    gt_x = bboxes[i, 0].item()
    gt_y = bboxes[i, 1].item()
    gt_h = bboxes[i, 2].item()
    gt_w = bboxes[i, 3].item()

    iou = calculate_iou(
        [pred_x, pred_y, pred_w, pred_h],
        [gt_x, gt_y, gt_w, gt_h]
    )

    # Class
    class_logits = pred[y, x, 10:13]
    pred_class = torch.argmax(class_logits).item()
    true_class = labels[i].item()

    # Compact diagnostic
    print(
        f"{i}: cell=({x},{y}) box={box_idx+1} "
        f"conf={confidence:.2f} IoU={iou:.2f} | "
        f"P=[{pred_x:.0f},{pred_y:.0f},{pred_w:.0f},{pred_h:.0f}] "
        f"GT=[{gt_x:.0f},{gt_y:.0f},{gt_w:.0f},{gt_h:.0f}] | "
        f"{idx_to_class[pred_class]}/{idx_to_class[true_class]}"
    )

    # Image
    img = images[i] * std + mean
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()

    axes[i].imshow(img)

    # Prediction = green
    axes[i].add_patch(
        patches.Rectangle(
            (pred_x - pred_w/2, pred_y - pred_h/2),
            pred_w, pred_h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none"
        )
    )

    # Ground truth = red
    axes[i].add_patch(
        patches.Rectangle(
            (gt_x - gt_w/2, gt_y - gt_h/2),
            gt_w, gt_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--"
        )
    )

    axes[i].set_title(
        f"IoU={iou:.2f} | {idx_to_class[pred_class]}",
        fontsize=10
    )
    axes[i].axis("off")

plt.tight_layout()
plt.show()

model.eval()

images, bboxes, labels = next(iter(train_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)

outputs = outputs.permute(0, 2, 3, 1)

idx_to_class = {0:"cucumber", 1:"eggplant", 2:"mushroom"}

mean = torch.tensor([0.485,0.456,0.406], device=device).view(3,1,1)
std  = torch.tensor([0.229,0.224,0.225], device=device).view(3,1,1)

fig, axes = plt.subplots(4, 5, figsize=(18,14))
axes = axes.flatten()

for i in range(min(20, len(images))):

    pred = outputs[i]

    confs = torch.stack([
        torch.sigmoid(pred[...,0]),
        torch.sigmoid(pred[...,5])
    ], dim=-1)

    max_idx = torch.argmax(confs)

    y = (max_idx // 14).item()
    x = ((max_idx // 2) % 7).item()
    box_idx = (max_idx % 2).item()

    raw = pred[y,x,1:5] if box_idx == 0 else pred[y,x,6:10]

    bx = torch.sigmoid(raw[0]).item()
    by = torch.sigmoid(raw[1]).item()
    bh = torch.abs(raw[2]).item()
    bw = torch.abs(raw[3]).item()

    cell = Image_size / 7

    px = (x + bx) * cell
    py = (y + by) * cell
    pw = bw * Image_size
    ph = bh * Image_size

    gx = bboxes[i,0].item()
    gy = bboxes[i,1].item()
    gh = bboxes[i,2].item()
    gw = bboxes[i,3].item()

    # IoU
    px1, px2 = px-pw/2, px+pw/2
    py1, py2 = py-ph/2, py+ph/2
    gx1, gx2 = gx-gw/2, gx+gw/2
    gy1, gy2 = gy-gh/2, gy+gh/2

    inter = max(0,min(px2,gx2)-max(px1,gx1)) * \
            max(0,min(py2,gy2)-max(py1,gy1))

    iou = inter / (pw*ph + gw*gh - inter + 1e-6)

    pred_class = torch.argmax(pred[y,x,10:13]).item()
    true_class = labels[i].item()

    # Unnormalize
    img = images[i] * std + mean
    img = img.clamp(0,1).permute(1,2,0).cpu().numpy()

    axes[i].imshow(img)

    # Prediction = green
    axes[i].add_patch(
        patches.Rectangle(
            (px-pw/2, py-ph/2), pw, ph,
            fill=False, linewidth=2, edgecolor="lime"
        )
    )

    # Ground truth = red dashed
    axes[i].add_patch(
        patches.Rectangle(
            (gx-gw/2, gy-gh/2), gw, gh,
            fill=False, linewidth=2,
            edgecolor="red", linestyle="--"
        )
    )

    axes[i].set_title(
        f"IoU={iou:.2f} | "
        f"P:{idx_to_class[pred_class]} "
        f"GT:{idx_to_class[true_class]}"
    )

    axes[i].axis("off")

plt.tight_layout()
plt.show()


