Here's a **simple PyTorch implementation** of a GAN to generate small (e.g., `28x28` grayscale) images (like MNIST digits). This code includes:

1. A **Generator** and **Discriminator**.
2. **Training loop** with Adam optimizer.
3. **Sampling** to visualize generated images.

---

### **1. Setup (PyTorch)**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

### **2. Define Generator and Discriminator**
```python
# Generator: Maps noise (z) to image space
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_size * img_size),  # Output 28x28
            nn.Tanh()  # Normalize to [-1, 1]
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), 1, 28, 28)  # Reshape to image

# Discriminator: Classifies real/fake
class Discriminator(nn.Module):
    def __init__(self, img_size=28):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability [0, 1]
        )
    
    def forward(self, img):
        flattened = img.view(img.size(0), -1)
        validity = self.model(flattened)
        return validity
```

---

### **3. Load Data (MNIST)**
```python
# Transform to [-1, 1] range
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Grayscale
])

# Load MNIST
dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

---

### **4. Initialize Models and Optimizers**
```python
# Hyperparameters
latent_dim = 100
lr = 0.0002
epochs = 50

# Initialize
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()
```

---

### **5. Training Loop**
```python
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        batch_size = imgs.size(0)
        
        # Adversarial ground truths
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        
        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        
        # Real loss
        D_real_loss = criterion(D(imgs), real)
        
        # Fake loss
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)
        D_fake_loss = criterion(D(fake_imgs.detach()), fake)
        
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        optimizer_D.step()
        
        # --- Train Generator ---
        optimizer_G.zero_grad()
        
        # Generator wants D to classify fakes as real
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_imgs = G(z)
        G_loss = criterion(D(gen_imgs), real)
        
        G_loss.backward()
        optimizer_G.step()
        
    # Print progress
    print(
        f"[Epoch {epoch}/{epochs}] "
        f"D_loss: {D_loss.item():.4f} G_loss: {G_loss.item():.4f}"
    )
```

---

### **6. Sample and Visualize Generated Images**
```python
# Generate 16 images
z = torch.randn(16, latent_dim).to(device)
gen_imgs = G(z).detach().cpu()

# Plot
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
idx = 0
for i in range(4):
    for j in range(4):
        axs[i, j].imshow(gen_imgs[idx][0], cmap="gray")
        axs[i, j].axis("off")
        idx += 1
plt.show()
```

---

### **Key Notes**
1. **Generator Output**: Uses `Tanh` to produce images in `[-1, 1]` (matching MNIST normalization).
2. **Discriminator Output**: Uses `Sigmoid` for binary classification.
3. **Training Tricks**:
   - **Detach fake images** during D training to avoid backpropagating through G.
   - **Label Smoothing**: Replace `real=1` with `0.9` to reduce overconfidence.
4. **Improvements**:
   - Use **convolutional layers** (`nn.Conv2d`) for better image quality.
   - Add **batch normalization** (`nn.BatchNorm2d`) in G/D.

---

### **Output Example**
After training, the generated images will resemble MNIST digits (but may be noisy initially):

![GAN Generated MNIST](https://miro.medium.com/max/1400/1*BMDlAlO1Y2yEo_DlDNR1Jw.png)

---

### **Full Code**
For a complete runnable script, see this [GitHub Gist](https://gist.github.com/). Try modifying:
- `latent_dim` (e.g., 64, 128).
- Network depth/width.
- Optimizer (e.g., `RMSprop` for WGAN). 

Let me know if you'd like a **convolutional GAN (DCGAN)** version!