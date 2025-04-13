import numpy as np
import matplotlib.pyplot as plt
# here we define a perceptron using PyTorch


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NetN1(nn.Module):
    def __init__(self):
        super(NetN1, self).__init__()
        self.fc1 = nn.Linear(2, 1)
        

    def forward(self, x):
        x = self.fc1(x)
        return x




def TrainMyNet(fileName):
    # 1. Load the data
    D = np.loadtxt(fileName)
    # 2. separate data nd labels
    X = D[:,0:2]
    C = D[:,2]
    """
        train your perceptron here and save the trained net like this:
        torch.save(net, 'myPerceptron1.pth')
        
        don't forget to normalize the data!
        
    """
    

def TestMyNet(fileName):
    # 1. Load the data
    D = np.loadtxt(fileName)
    # 2. separate data nd labels
    X = D[:,0:2]
    C = D[:,2]
    """
        load the test data and the saved net and classify the data
        
        don't forget to use the normalization info
        
    """

    
task = 'test'  #'train'

if task == 'train':
    fname = "trainSet2.txt"   #input('Enter train file name:')
    TrainMyNet(fname)
    print('done training')
    
    
if task == 'test':
    fname =  "testSet2.txt" #input('Enter test file name:')
    TestMyNet(fname)
    print('done training')    