import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
        root = "data",
        train = True,
        download=True,
        transform=ToTensor()
        )
test_data = datasets.FashionMNIST(
        root = "data",
        train = False,
        download=True,
        transform=ToTensor()
        )

train_dataloader = DataLoader(training_data,batch_size=64)
test_dataloader = DataLoader(training_data,batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() #Es una herencia
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28,512),
                nn.ReLU(),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512,10),
                )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def train_loop(dataloader,model,loss_fn,optimizer):
        size = len(dataloader.dataset)
        #Set te model to training mode - important for batch normalization 
        #Unnecessary on this situaion but added for best practices
        model.train()
        for batch, (X,y) in enumerate(dataloader):
            #Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred,y)
    
            #Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
            if batch % 100 == 0:
                loss, current = loss.item(), batch * size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader,model,loss_fn):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0,0
        with torch.no_grad():


model = NeuralNetwork()
