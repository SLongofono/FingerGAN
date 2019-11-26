import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

# path to training images
dataroot = "data"

# training batch size
batch_size = 128

# image dimension pixels
image_size = 64

# color channels
nc = 3

# latent vector size for generator input
nz = 100

# feature size generator
ngf = 64

# feature size discriminator
ndf = 64

# training runs ("epochs")
num_epochs = 50

# learning rate, from paper
lr = 0.0002

# param for Adam optimizer
beta1 = 0.5

# initializes weights per DCGAN paper
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
    Helper class for the generator module

    We have four convolutional stages, with normalizations between each
    Individual kernels are 4x4, with other dimensions/strides scaled
    accordingly.

    The rectified linear activation function ReLU allows fast training while
    avoiding problems associated with sigmoids, etc (vanishing gradients)
    The final application of tanh maps the output to [-1, 1] like our
    normalized input images.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input to convolution stage
            # intermediate state (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # intermediate state (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # intermediate state (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # intermediate state ngff x 32 x 32
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # output state 64 x 64
        )

    def forward(self, inp):
        """
        Feed input data forward to produce an output image
        """
        return self.main(inp)


class Discriminator(nn.Module):
    """
    helper class for the discriminator module

    This operates similarly to the generator, except that now we are
    convolving with the current weights working toward a value judgement at
    the end.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input size 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # intermediate state ndf x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # intermediate state (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # intermediate state (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # intermediate state (ndf*8) x 4 x 4
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        """
        Feed input data forward to produce an output classification
        """
        return self.main(inp)

# Data marshalling helpers
dataset = dset.ImageFolder(root=dataroot,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
shuffle=True, num_workers=2)

device = torch.device("cpu")

# Sanity check on dataset
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],padding=2,
normalize=True).cpu(),(1,2,0)))
plt.show()

# Set up networks
generator = Generator().to(device)
generator.apply(init_weights)
print(generator)

discriminator = Discriminator().to(device)
discriminator.apply(init_weights)
print(discriminator)

# Set up feedback optimizers
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG= optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))


# Training loop

img = []
G_loss = []
D_loss = []
iters = 0

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Train with known good data

        # zero out gradients
        discriminator.zero_grad()

        # prepare a batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)

        # feed forward
        output = discriminator(real_cpu).view(-1)

        # compute error and backpropagate
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with known fake data
        # prepare a batch of random data for the latent space
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # generate batch of fake images
        fake = generator(noise)
        label.fill_(fake_label)

        # feed forward
        output = discriminator(fake.detach()).view(-1)

        #compute error and backpropagate
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake

        optimizerD.step()

        # Train G

        # prepare a batch of random data for the latent space
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # generate batch of fake images
        fake = generator(noise)

        # zero out gradients
        generator.zero_grad()

        label.fill_(real_label)

        # get feedback from discriminator
        output = discriminator(fake).view(-1)

        #compute error and backpropagate
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()

        optimizerG.step()

        if i % 25 == 0:
            print('Epoch [%d/%d] Batch [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'% (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_loss.append(errG.item())
        D_loss.append(errD.item())

        if (iters % 10 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img.append(vutils.make_grid(fake, padding=2, normalize=True))
            print(fake.shape)
            im_tens = fake[0]
            vutils.save_image(im_tens, f"gen_{iters}.bmp")

        iters += 1

# plot loss over time
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss")
plt.plot(G_loss, label='G')
plt.plot(D_loss, label='D')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()
plt.show()

# animate images development in generator
#fig  = plt.figure(figsize=(8,8))
#plt.axis("off")
#ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img]
#ani = animation.ArtistAnimation(fig, ims, interval=1000,repeat_delay=1000,blit=True)
#HTML(ani.to_jshtml())


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img[-1],(1,2,0)))
plt.show()

# save the optimizer and discriminator state for the demo
torch.save(
    {
        'discriminator_state_dict':discriminator.state_dict(),
        'optimizerD_state_dict':optimizerD.state_dict(),
        'generator_state_dict':generator.state_dict(),
        'optimizerG_state_dict':optimizerG.state_dict()
    },
    'networkStates.pyt'
)

