import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image

# 1. Device and Image Paths Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

style_path = "C:/Users/Selvakkumaran S V/Downloads/Starry Night Van Gogh.png"
content_path = "C:/Users/Selvakkumaran S V/Downloads/Mona_Lisa.PNG"

# Define image size (smaller sizes train faster, e.g., 256 or 512)
imsize = 256 if torch.cuda.is_available() else 128

# 2. Preprocessing Pipeline
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# Load images
content_image = load_image(content_path)
style_image = load_image(style_path)

# Initialize generated image as a clone of the content image
generated_image = content_image.clone().requires_grad_(True)

# 3. Load VGG-19 and Freeze Parameters
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device)
print(vgg)
for param in vgg.parameters():
    param.requires_grad = False


# 4. Loss Functions & Gram Matrix
def compute_content_loss(gen_features, content_features):
    return torch.mean((gen_features - content_features) ** 2)


def gram_matrix(tensor):
    batch_size, channels, height, width = tensor.size()
    features = tensor.view(channels, height * width)
    gram = torch.mm(features, features.t())
    return gram / (channels * height * width)


def compute_style_loss(gen_gram, style_gram):
    return torch.mean((gen_gram - style_gram) ** 2)


# 5. Feature Extraction Helper
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


content_layer = {'21': 'conv4_2'}
style_layers = {
    '0': 'conv1_1',
    '5': 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '28': 'conv5_1'
}
all_layers = {**content_layer, **style_layers}

# Extract target features
content_features = get_features(content_image, vgg, all_layers)
style_features = get_features(style_image, vgg, all_layers)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers.values()}

# Hyperparameters & Optimizer
content_weight = 1e1
style_weight = 5e5
optimizer = optim.LBFGS([generated_image], lr=1.0)

# 6. Optimization Loop
epochs = 300
run = [0]

print("Starting Neural Style Transfer optimization...")
while run[0] <= epochs:
    def closure():
        optimizer.zero_grad()
        gen_features = get_features(generated_image, vgg, all_layers)

        c_loss = compute_content_loss(gen_features['conv4_2'], content_features['conv4_2'])

        s_loss = 0
        for layer in style_layers.values():
            gen_gram = gram_matrix(gen_features[layer])
            s_gram = style_grams[layer]
            s_loss += compute_style_loss(gen_gram, s_gram) / len(style_layers)

        total_loss = content_weight * c_loss + style_weight * s_loss
        total_loss.backward()

        if run[0] % 50 == 0:
            print(f"Epoch {run[0]} / {epochs} | Content Loss: {c_loss.item():.4f} | Style Loss: {s_loss.item():.4f}")

        run[0] += 1
        return total_loss


    optimizer.step(closure)


# 7. Save Output
def save_output(tensor, output_path):
    image = tensor.cpu().clone().detach().squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    pil_image = TF.to_pil_image(image)
    pil_image.save(output_path)
    print(f"Saved styled image to {output_path}")


save_output(generated_image, "output_styled_image.jpg")