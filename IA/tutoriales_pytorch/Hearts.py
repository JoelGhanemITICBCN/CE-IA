import csv
import torch 
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

file_path = "heart_disease.csv"
data = []
with open(file_path,'r') as file:
    reader = csv.reader(file)
