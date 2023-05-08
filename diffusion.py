import math
from time import sleep
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import torchvision.datasets as datasets
import torch.utils.data.dataloader as dataloader
import matplotlib.pyplot as plt




def show_images(dataset,num_samples,cols):
    plt.figure(figsize=(15,15))
    for i,img in enumerate(dataset):
        if i ==  num_samples:
            break
        plt.subplot(num_samples//cols + 1 , cols, i+1)
        plt.imshow(img[0])
    plt.show()

def get_beta(timestep,start=0.0001,end=0.01):
    return torch.linspace(start,end,timestep)




def get_index_from_list(vals,t,x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1,t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
def forward_diffusion_sample(x_0,t,device="cuda"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

T = 300
betas = get_beta(timestep=T)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas,axis=0)
alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1],(1,0),value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0/alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1 - alphas_cumprod)

IMG_SIZE = 256
BATCH_SIZE = 16

def dataload():
    data_transforms=[
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t:2*t-1)
    ]
    data_transform= transforms.Compose(data_transforms)

    train = datasets.Flowers102(root="./data",download=False,split="train",transform=data_transform)
    test = datasets.Flowers102(root="./data",download=False,split="test",transform=data_transform)
    val = datasets.Flowers102(root="./data",download=False,split="val",transform=data_transform)

    dataset=torch.utils.data.ConcatDataset([train,test,val])
    print(len(dataset))
    train, test = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    return torch.utils.data.ConcatDataset([train,test])

def show_tensor_image(image):
    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda t:(t+1)/2),
        transforms.Lambda(lambda t:t.permute(1,2,0)),
        transforms.Lambda(lambda t:t*255.),
        transforms.Lambda(lambda t:t.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0,:,:,:]
    plt.imshow(reverse_transform(image))

class Block(nn.Module):
    def __init__(self,in_ch,out_ch,time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim,out_ch)
        if up :
            self.conv1 = nn.Conv2d(2*in_ch,out_ch,3,padding=1)
            self.transform = nn.ConvTranspose2d(out_ch,out_ch,4,2,1)
        else:
            self.conv1 = nn.Conv2d(in_ch,out_ch,3,padding=1)
            self.transform = nn.Conv2d(out_ch,out_ch,4,2,1)
        self.conv2= nn.Conv2d(out_ch,out_ch,3,padding=1)
        self.pool = nn.MaxPool2d(3,stride=2)
        self.bnorm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self,x,t,):
        h = self.bnorm(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, )*2]
        h = h + time_emb
        h = self.bnorm(self.relu(self.conv2(h)))
        return self.transform(h)

class SinusoidalPositionEmbbedings(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
    def forward(self,time):
        device = time.device
        half_dim = self.dim//2
        embbedings = math.log(10000) / (half_dim-1)
        embbedings = torch.exp(torch.arange(half_dim,device = device)*-embbedings)
        embbedings = time[:, None] * embbedings[None , :]
        embbedings = torch.cat((embbedings.sin(),embbedings.cos()),dim=-1)
        return embbedings

class SimpleUnet(nn.Module):

    def __init__(self):
        super().__init__()
        image_channels = 3  
        down_channels = (64,128,256,512,1024)
        up_channels = (1024,512,256,128,64)
        out_dim = 3
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbbedings(time_emb_dim),
            nn.Linear(time_emb_dim,time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels,down_channels[0],3,padding=1)
        self.downs = nn.ModuleList([Block(down_channels[i],down_channels[i+1],time_emb_dim) for i in range(len(down_channels)-1)])
        self.ups = nn.ModuleList([Block(up_channels[i],up_channels[i+1],time_emb_dim,up=True) for i in range(len(up_channels)-1)])
        self.output = nn.Conv2d(up_channels[-1],out_dim,1)

    def forward(self,x,timestep):
        t=self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs=[]
        for down in self.downs:
            x = down(x,t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x,residual_x),dim = 1)
            x = up(x,t)
        x=self.output(x)
        return x

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device="cuda")
    noise_pred = model(x_noisy, t)
    return torch.nn.functional.l1_loss(noise, noise_pred)
@torch.no_grad()
def sample_timestep(x,t):
    betas_t = get_index_from_list(betas,t,x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod,t,x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas,t,x.shape)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x,t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance,t,x.shape)
    if t == 0 :
        return model_mean
    else :
        noise = torch.randn_like(x)
        return model_mean+torch.sqrt(posterior_variance_t)*noise
    
@torch.no_grad()
def sample_plot_image():
    img_size = IMG_SIZE
    img = torch.randn((1,3,img_size,img_size),device = "cuda")
    plt.figure(figsize=(15,15))
    plt.axis("off")
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,),i,device = "cuda",dtype = torch.long)
        img = sample_timestep(img,t)
        if i%stepsize == 0:
            plt.subplot(1,num_images,int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())


if __name__ == "__main__":
    data = dataload()
    data_loader = dataloader.DataLoader(data,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

    image = next(iter(data_loader))[0]

    plt.figure(figsize=(15,15))
    plt.axis("off")
    num_images = 10
    stepsize = int(T/num_images)

    for idx in range(0,T,stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1,num_images+1,int(idx / stepsize)+ 1)
        image , noise = forward_diffusion_sample(image,t)
        show_tensor_image(image)
    plt.savefig(f"./diffusion_img/forward.png")


    model = SimpleUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    print(model)

    from torch.optim import Adam

    model.to("cuda")
    opt = Adam(model.parameters(),lr = 0.001)
    epochs = 100
    for epoch in range(epochs):
        for step,batch in enumerate(data_loader):
            print(f"step {step}")
            
            opt.zero_grad()

            t = torch.randint(0,T,(BATCH_SIZE,),device="cuda").long()
            loss = get_loss(model,batch[0],t)
            loss.backward()
            opt.step()

            if step%50 == 0:
                print(f"Epoch {epoch}|Step {step:03d} Loss:{loss.item()} ")
                plt.close('all')
                sample_plot_image()
                plt.savefig(f"./diffusion_img/{epoch:03d}-{step:03d}.png")
                print("img saved")
    torch.save(model.state_dict(),"diffusion_model.pth")




