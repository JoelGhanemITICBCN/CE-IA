import torch
import numpy as np

# Iniciar Tensor default
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

#Iniciar tensor a partir de Numpy Array
np_array  = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

#Premade formats
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random \n{rand_tensor} \n")
print(f"Ones \n{ones_tensor} \n")
print(f"Zeros \n{zeros_tensor} \n")

#Attributes

tensor = torch.rand(3,4)
print(f"Shape :{tensor.shape}")
print(f"Datatype :{tensor.dtype}")
print(f"device :{tensor.device}")

#Operations on Tensors
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4,4)
print(f"First row:{tensor[0]}")
print(f"First column:{tensor[:,0]}")
print(f"Last column:{tensor[...,-1]}")
tensor[:,1] = 0
print(tensor)

#Concat
t1 = torch.cat([tensor,tensor,tensor],dim=1)
print(t1)

#Arithmetic operations
y1 = tensor@tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
print("y3")
antorcha = torch.matmul(tensor,tensor.T,out=y3)
print(antorcha)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
print("z3")

antorcha = torch.mul(tensor,tensor,out=z3)
print(antorcha)

#Aggregations
print("Aggregations")
agg = tensor.sum()
agg_item = agg.item()
print(agg_item,type(agg_item))

print(f"{tensor} \n")
tensor.add_(5) #Adds 5 to every value
print(tensor)

#Bridge with Numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

#Numpy to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n,1,out=n)
print(f"t: {t}")
print(f"n: {n}")
