import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
latent_dim = 100
img_channels = 1
img_size = 28
batch_size = 64
lr = 0.0002
epochs = 50

# Generator: Maps noise to fake images
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: noise (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x):
        return self.main(x)

# Discriminator: Classifies real vs. fake
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: image (img_channels, img_size, img_size)
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        return self.main(x).view(-1)

torch.manual_seed(42)

# Initialize networks
generator = Generator()
discriminator = Discriminator()

# Optimizers
opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
])
dataset = torchvision.datasets.MNIST(
    root="./data", transform=transform, download=True
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        # Train Discriminator
        discriminator.zero_grad()
        
        # Real images
        real_labels = torch.ones(real_imgs.size(0))
        real_output = discriminator(real_imgs)
        loss_real = criterion(real_output, real_labels)
        
        # Fake images
        noise = torch.randn(real_imgs.size(0), latent_dim, 1, 1)
        fake_imgs = generator(noise)
        fake_labels = torch.zeros(real_imgs.size(0))
        fake_output = discriminator(fake_imgs.detach())
        loss_fake = criterion(fake_output, fake_labels)
        
        loss_d = loss_real + loss_fake
        loss_d.backward()
        opt_d.step()

        # Train Generator
        generator.zero_grad()
        fake_output = discriminator(fake_imgs)
        loss_g = criterion(fake_output, real_labels)  # Trick: use real_labels here
        loss_g.backward()
        opt_g.step()

    # Print progress
    print(f"Epoch [{epoch}/{epochs}] | D_loss: {loss_d.item():.4f} | G_loss: {loss_g.item():.4f}")

    # Save generated images
    if epoch % 10 == 0:
        with torch.no_grad():
            sample_noise = torch.randn(16, latent_dim, 1, 1)
            generated = generator(sample_noise)
            plt.figure(figsize=(4, 4))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(generated[i].squeeze(), cmap='gray')
                plt.axis('off')
            plt.show()

"""
Epoch [0/50] | D_loss: 2.7096 | G_loss: 3.8826
Epoch [1/50] | D_loss: 0.6798 | G_loss: 3.3860
Epoch [2/50] | D_loss: 0.3155 | G_loss: 1.4586
"""