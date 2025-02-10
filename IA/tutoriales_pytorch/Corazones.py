import csv
import torch
import numpy as np
from torch.autograd import backward
import torch.nn as nn
from torch.nn.modules import ReLU
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Leer el CSV manualmente
file_path = "heart_disease.csv"
data = []

#Abrir y leer el csv 
data = np.genfromtxt(file_path,delimiter=',') #Especifico archivo y como separarlo
data = np.delete(data,0,0) #Borra el encabezado
data_x = torch.tensor(np.delete(data,(-1),1),dtype=torch.float32) #Borro la ultima columna
data_y = torch.tensor(np.delete(data,slice(0,-1),1),dtype=torch.float32) #Borro todo menos la ultima columna

# torch.tensor({algo},dtype=float32) -> Convertir datos a tensores

# Separar características (X) y etiquetas (y)
X = data_x
y = data_y

#Creamos el Dataset
class HeartDiseaseDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets.unsqueeze(1)  # Asegurar que sea un vector columna
        #unsqueeze() Hace que sea (N,1) para asegurarse de que los tensores van bien
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

#Crear el Dataset Completo 
dataset = HeartDiseaseDataset(X, y)

total_size = len(dataset)
train_size = total_size * 0.8
test_size = total_size - train_size

# Dividir en conjuntos de entrenamiento y prueba
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,test_size])

batch_size = 64 

# Escalar las características
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

# Crear un conjunto de datos personalizado

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Definir la red neuronal
class HeartDiseaseNN(nn.Module):
    def __init__(self):
        super(HeartDiseaseNN, self).__init__()
        self.model_stack = torch.nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
            )

    def forward(self, X):
        return self.model_stack(X)

# Instanciar el modelo, la función de pérdida y el optimizador
model = HeartDiseaseNN()

criterion = nn.CrossEntropyLoss()  # Binary Cross Entropy para clasificación binaria
optimizer = optim.Adam(model.parameters(), lr=0.03)


def train_loop(model, dataloader, loss, optimizer):

    for batch, (X,y) in dataloader:

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(model, dataloader):

    for batch, (X,y) in dataloader:
        logits = model(X)
        pred = nn.functional.softmax(logits)

        print(accuracy)




# Entrenar el modelo
epochs = 10
for epoch in range(epochs):
    train_loop()
    test_loop()

model.save()

