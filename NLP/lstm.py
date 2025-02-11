import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchrl.data import TensorDictTokenizer
from torchtune.models.mistral import MistralTokenizer
from transformers import AutoTokenizer
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable


epochs = 100
batch_size = 32
learning_rate = 0.00001
embedding_dim = 100
hidden_dim = 200
label_size = 2 

#Using bert's uncased tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

class BertClassifier(nn.Module):
    def __init__(self, d_model, vocab_size, label_size=2, nhead=8, nencoders=6):
        super(BertClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nencoders)
        self.hidden2label = nn.Linear(102400, label_size)


    def forward(self, sentence):
        embeds = self.embedding(sentence)
        encoder_out= self.encoder(embeds)
        encoder_out = encoder_out.reshape(encoder_out.shape[0],encoder_out.shape[1]*encoder_out.shape[2] )
        logits  = self.hidden2label(encoder_out)
        return logits

class CustomDataset(Dataset):
    def __init__(self,csv,tokenizer):
        df = pd.read_csv(csv)
        df = df.replace("positive",1)
        df = df.replace("negative",0)
        self.x = df["review"]
        self.y = df["sentiment"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        sentence = self.x[idx]

        tokens = self.tokenizer.encode(sentence, max_length=512, truncation=True)
        padding_len = 512 - len(tokens)
        padded_tokens = tokens + [0]*padding_len
        label = self.y[idx]

        return torch.tensor(padded_tokens, dtype=torch.long),label

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            #print(pred.shape)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            #Compiling results
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    test_loss /= num_batches
    correct /= size
    for true_label, predicted in zip(all_labels,all_preds):
        print(f"True label: {true_label}, Predicted: {predicted}")
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")
device = "cuda"


dataset = CustomDataset("IMDBDataset.csv",tokenizer)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = BertClassifier(d_model=200, vocab_size=tokenizer.vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer, device)
    test_loop(test_loader, model, loss_fn, device)
print("Done!")

