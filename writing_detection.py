from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable


class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2,),# stride = 1, padding = (kernel_size-1)/2 = (5-1)/2
        nn.ReLU(),# (16, 28, 28)
        nn.MaxPool2d(kernel_size = 2),# (16, 14, 14)
    )
    self.conv2 = nn.Sequential(# (16, 14, 14)
        nn.Conv2d(16, 32, 5, 1, 2),# (32, 14, 14)
        nn.ReLU(),# (32,14,14)
        nn.MaxPool2d(2)# (32, 7, 7)
    )
    self.out = nn.Linear(32*7*7, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    output = self.out(x)
    return output



if __name__=="__main__":
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=False,
        transform=ToTensor(),
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor(),
    )
    BATCH_SIZE=16
    device="cuda"
    training_dataloader= DataLoader(training_data,batch_size=BATCH_SIZE)
    test_dataloader= DataLoader(test_data,batch_size=BATCH_SIZE)
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break



    model = CNN().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Compute prediction error
            pred = model(X)
            # print(pred)

            # print(pred.size(),y.size())
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(),"model.pth")

# print("Use model")
# test_x = torch.unsqueeze(test_data.data, dim = 1).type(torch.cuda.FloatTensor)[:2000]/255.
# test_y = test_data.targets[:2000]
# test_output= model(test_x[:100])
# pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
# plt.ion()
# for i in range(11):
#     plt.imshow(test_data.data[i].numpy(),cmap='gray')
#     plt.title("Predict:{} Real:{}".format(pred_y[i],test_y[i]))
#     plt.pause(1.5)
    # print(pred_y, 'prediction number')
    # print(test_y[:10].numpy(), 'real number')