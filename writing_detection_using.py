from random import shuffle
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from writing_detection import CNN

BATCH_SIZE=64
test_data = datasets.MNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
)
test_dataloader= DataLoader(test_data,batch_size=BATCH_SIZE)
model=CNN().to("cuda")
model.load_state_dict(torch.load('model.pth'))
print(test_data.data[0])
test_x = torch.unsqueeze(test_data.data, dim = 1).type(torch.cuda.FloatTensor)[:2000]/255.

test_y = test_data.targets
# print(test_x)
# print(test_y)
test_output= model(test_x)
pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
plt.ion()
li=[i for i in range(2000)]
shuffle(li)
for i in li[:100]:
    plt.imshow(test_data.data[i].numpy(),cmap='gray')
    plt.title("Predict:{} Real:{}".format(pred_y[i],test_y[i]))
    plt.pause(1.5)
    # print(pred_y, 'prediction number')
    # print(test_y[:10].numpy(), 'real number')
    
