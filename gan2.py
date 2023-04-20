import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchvision.utils import save_image
training_data = datasets.MNIST(
        root="data",
        train=True,
        download=False,
        transform=ToTensor(),
    )
dataloader=DataLoader(training_data,batch_size=64)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2,),# stride = 1, padding = (kernel_size-1)/2 = (5-1)/2
            nn.ReLU(),# (16, 28, 28)
            nn.MaxPool2d(kernel_size = 2),# (16, 14, 14)
            nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(# (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),# (32, 14, 14)
            nn.ReLU(),# (32,14,14)
            nn.MaxPool2d(2),# (32, 7, 7)
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(32*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("x",x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2,),# stride = 1, padding = (kernel_size-1)/2 = (5-1)/2
            nn.ReLU(),# (16, 28, 28)
            nn.MaxPool2d(kernel_size = 2),# (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,8,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.layer= nn.Sequential(
            nn.Linear(8*7*7,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,1*28*28),
            nn.Tanh()
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        # print("g",x.size())
        x = self.layer(x)
        x = x.view(x.size(0),1,28,28)
        return x

discriminator=Discriminator().to("cuda")
generator=Generator().to("cuda")

loss_fn = nn.BCELoss()
d_opt = torch.optim.Adam(discriminator.parameters(),lr=0.0002)
g_opt = torch.optim.Adam(generator.parameters(),lr=0.0002)

EPOCH = 50

for epoch in range(EPOCH):
    for i,(images,_) in enumerate(dataloader):
        real_img = images.to("cuda")
        # print(real_img.size())
        
        

        discriminator.zero_grad()
        output_real=discriminator(real_img)
        real_labels = torch.ones(output_real.size(0),1).to("cuda")
        # print(output_real.size(),real_labels.size())
        loss_real=loss_fn(output_real,real_labels)

        z = torch.randn(64,1,28,28, device="cuda")
        fake_data=generator(z)
        output_fake = discriminator(fake_data)
        fake_labels = torch.zeros(output_fake.size(0),1).to("cuda")
        loss_fake=loss_fn(output_fake,fake_labels)

        loss_d = loss_fake + loss_real
        loss_d.backward(retain_graph=True)
        d_opt.step()

        generator.zero_grad()
        output_fake= discriminator(fake_data)
        real_labels = torch.ones(output_fake.size(0),1).to("cuda")
        loss_g = loss_fn(output_fake,real_labels)
        loss_g.backward()
        g_opt.step()

        if i%100 == 0 :
            save_image(fake_data[:25], f'./output_img/epoch_{epoch}batch{i}.png', nrow=5, normalize=True)
            print(f'Epoch [{epoch+1}/{EPOCH}] complete')
    print('Training finished.')