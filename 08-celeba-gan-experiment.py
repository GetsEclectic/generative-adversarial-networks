import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm
from apex import amp
from torchvision.utils import save_image
from shutil import rmtree

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "/workspace/imgs"
checkpoint_file_path = './checkpoint.tar'

# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr_d = 2e-4
lr_g = 5e-5
betas = (0.0, 0.999)
d_steps_per_g_step = 2

# decay for exponential moving average
ema_decay = 0.9999

# Number of GPUs available. Use 0 for CPU mode.
ngpu = torch.cuda.device_count()

#%%

# load checkpoint data if it exists
have_checkpoint = False
checkpoint = None
completed_epochs = 0
if os.path.exists(checkpoint_file_path):
    checkpoint = torch.load(checkpoint_file_path)
    have_checkpoint = True
    completed_epochs = checkpoint['completed_epochs']

#%%

# calculate frechet inception distance
num_batches = 200
real_images_for_fid_dir = "./fid_images/real/"
fake_images_for_fid_dir = "./fid_images/fake/"


def print_frechet_inception_distance():
    torch.cuda.empty_cache()
    rmtree(real_images_for_fid_dir, ignore_errors=True)
    rmtree(fake_images_for_fid_dir, ignore_errors=True)
    os.makedirs(real_images_for_fid_dir)
    os.makedirs(fake_images_for_fid_dir)

    real_images = torch.cat([x[0] for _, x in zip(range(num_batches), iter(dataloader))])

    for i in range(real_images.size(0)):
        save_image(real_images[i, :, :, :], real_images_for_fid_dir + '{}.png'.format(i))

    with torch.no_grad():
        for batch_num in range(num_batches):
            generator_inputs = torch.randn(batch_size, nz).to(device)
            fake_images = netG(generator_inputs).detach().cpu()
            for i in range(fake_images.size(0)):
                save_image(fake_images[i, :, :, :], fake_images_for_fid_dir + '{}.png'.format(i + batch_num * batch_size))

    # TODO: fix me
    # ! pytorch-fid/fid_score.py --batch-size 128 --gpu 0 ./fid_images/real/ ./fid_images/fake/

#%%

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

#%%

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#%%

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1))
        self.key_conv = spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1))
        self.value_conv = spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out

#%%

# Generator Code
class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.skip_conv = spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False))

        self.main_conv1 = spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, bias=False))
        self.main_bn = nn.BatchNorm2d(self.out_channels)
        self.main_relu = nn.ReLU(True)
        self.main_conv2 = spectral_norm(nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1, bias=False))

        self.out_bn = nn.BatchNorm2d(self.out_channels)
        self.out_relu = nn.ReLU(True)


    def forward(self, input):
        main = input
        skip = input

        skip = nn.functional.interpolate(skip, scale_factor=2)
        skip = self.skip_conv(skip)

        main = nn.functional.interpolate(main, scale_factor=2)
        main = self.main_conv1(main)
        main = self.main_bn(main)
        main = self.main_relu(main)
        main = self.main_conv2(main)

        out = main + skip
        out = self.out_bn(out)
        out = self.out_relu(out)
        return out


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.linear = nn.Sequential(
            spectral_norm(nn.Linear(nz, 4 * 4 * ngf * 16))
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            ResBlockUp(ngf * 16, ngf * 8),
            ResBlockUp(ngf * 8, ngf * 4),
            ResBlockUp(ngf * 4, ngf * 2),
            Self_Attn(ngf * 2, 'relu'),
            ResBlockUp(ngf * 2, ngf),
            spectral_norm(nn.Conv2d( ngf, nc, 3, 1, 1, bias=False)),
            nn.Tanh()
        )


    def forward(self, input):
        out = self.linear(input)
        out = out.view(-1, 1024, 4, 4)
        out = self.main(out)
        return out

#%%

# Create the generator
netG = Generator(ngpu).to(device)

# # Create EMA generator
# netG_ema = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

if not have_checkpoint:
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)
else:
    netG.load_state_dict(checkpoint['netG_state_dict'])
#   netG_ema.load_state_dict(checkpoint['netG_ema_state_dict'])

# Print the model
print(netG)

#%%

class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.skip_conv = spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0))

        self.main_conv1 = spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1))
        self.main_relu = nn.ReLU(True)
        self.main_conv2 = spectral_norm(nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1))

        self.out_relu = nn.ReLU(True)


    def forward(self, input):
        main = input
        skip = input

        skip = self.skip_conv(skip)
        skip = nn.functional.avg_pool2d(skip, 2)

        main = self.main_conv1(main)
        main = self.main_relu(main)
        main = self.main_conv2(main)
        main = nn.functional.avg_pool2d(main, 2)

        out = main + skip
        out = self.out_relu(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            ResBlockDown(nc, ndf),
            Self_Attn(ndf, 'relu'),
            # state size. (ndf) x 32 x 32
            ResBlockDown(ndf, ndf * 2),
            # state size. (ndf*2) x 16 x 16
            ResBlockDown(ndf * 2, ndf * 4),
            # state size. (ndf*4) x 8 x 8
            ResBlockDown(ndf * 4, ndf * 8),
            # state size. (ndf*8) x 4 x 4
        )

        self.linear = nn.Sequential(
            spectral_norm(nn.Linear(ndf * 8, 1))
        )

    def forward(self, input):
        out = self.main(input)
        out = torch.sum(out.view(out.size(0), out.size(1), -1), dim=2)
        return self.linear(out)

#%%

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

if not have_checkpoint:
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
else:
    netD.load_state_dict(checkpoint['netD_state_dict'])

# Print the model
print(netD)

#%%

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=betas)
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=betas)

# mixed precision initialization
netD, optimizerD = amp.initialize(
    netD, optimizerD, opt_level="O2",
    keep_batchnorm_fp32=True, loss_scale="dynamic"
)

netG, optimizerG = amp.initialize(
    netG, optimizerG, opt_level="O2",
    keep_batchnorm_fp32=True, loss_scale="dynamic"
)

if have_checkpoint:
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    amp.load_state_dict(checkpoint['amp'])

#%%

# Training Loop
from datetime import datetime
import math

# Lists to keep track of progress
G_losses = []
D_losses = []
iters = 0

final_epoch = num_epochs + completed_epochs

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    start_time = datetime.now()

    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        for j in range(d_steps_per_g_step):
            netD.zero_grad()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            # Format batch
            real_cpu = data[0].to(device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = torch.nn.ReLU()(1.0 - output).mean()
            D_x = errD_real.item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz).to(device)
            # Generate fake image batch with G
            fake = netG(noise)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = torch.nn.ReLU()(1.0 + output).mean()
            D_G_z1 = errD_fake.item()

            errD = errD_fake + errD_real
            with amp.scale_loss(errD, optimizerD) as scaled_loss:
                scaled_loss.backward()

            # Update D
            optimizerD.step()

        netG.zero_grad()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = - output.mean()
        # Calculate gradients for G
        with amp.scale_loss(errG, optimizerG) as scaled_loss:
            scaled_loss.backward()
        D_G_z2 = errG.item()

        # Update G
        optimizerG.step()

        # update netG_ema
        #         with torch.no_grad():
        #             for key in netG.state_dict():
        #                 netG_ema.state_dict()[key].data.copy_(netG_ema.state_dict()[key] * ema_decay
        #                                                      + netG.state_dict()[key] * (1 - ema_decay))

        # Output training stats
        if i % math.ceil(len(dataloader) / 5) == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (1 + completed_epochs, final_epoch, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1

    # save models and optimizers
    completed_epochs += 1
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        #                 'netG_ema_state_dict': netG_ema.state_dict(),
        'completed_epochs': completed_epochs,
        'amp': amp.state_dict()
    }, checkpoint_file_path)

    print("training time: " + str(datetime.now() - start_time))

    if completed_epochs % 5 == 0:
        print_frechet_inception_distance()


# plot of D & G’s losses versus training iterations.

#%%

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%%

# plot real images and fake images side by side

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
with torch.no_grad():
    fake = netG(torch.randn(64, nz).to(device)).detach().cpu()

fakes = vutils.make_grid(fake, padding=2, normalize=True)

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(fakes,(1,2,0)))
plt.show()
