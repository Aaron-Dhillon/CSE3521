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

    # 3. Normalize the data
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    
    #4. Save normalization params
    np.savez('normalization_params.npz', mean=mean, std=std)

    #5. Pytorch dataset and DataLoader setup
    X_tens = torch.FloatTensor(X_normalized)
    C_tens = torch.FloatTensor(C).view(-1,1)
    
    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(X_tens, C_tens)
    
    # Create DataLoader
    batch_size = 32  # You can adjust this value
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    #6. Initialize network and optimizer
    net = NetN1()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    #7. Training loop with batches
    epochs = 1000
    for epoch in range(epochs):
        for batch_X, batch_C in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = net(batch_X)
            loss = criterion(outputs, batch_C)
            loss.backward()
            optimizer.step()
            
            
        
    
    #8. Save the trained network
    torch.save(net, 'myPerceptron1.pth')
    
    #9. Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=C, cmap=plt.cm.Paired, edgecolors='k')
    plt.title('Training Data with Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Create mesh grid for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Normalize mesh data
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    mesh_data_normalized = (mesh_data - mean) / std
    
    # Get predictions for mesh
    net.eval()
    with torch.no_grad():
        Z = net(torch.FloatTensor(mesh_data_normalized)).numpy().reshape(xx.shape)
    
    # Plot decision boundary (simplified)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='-', linewidths=2)
    
    # Save plot
    plt.savefig(f'decision_boundary_{fileName.split(".")[0]}.pdf')
    plt.close()
    

def TestMyNet(fileName):
    # 1. Load the data
    D = np.loadtxt(fileName)
    # 2. separate data nd labels
    X = D[:,0:2]
    C = D[:,2]
    
    # 3. Load the trained network
    net = torch.load('myPerceptron1.pth')
    net.eval()  # Set to evaluation mode
    
    # 4. Load normalization parameters
    norm_params = np.load('normalization_params.npz')
    mean = norm_params['mean']
    std = norm_params['std']
    
    # 5. Normalize test data using training parameters
    X_normalized = (X - mean) / std
    
    # 6. Convert to tensor
    X_tens = torch.FloatTensor(X_normalized)
    
    # 7. Make predictions
    with torch.no_grad():
        outputs = net(X_tens)
        predicted = torch.sign(outputs).flatten().numpy()
    
    # 8. Calculate and print accuracy
    accuracy = np.mean(predicted == C) * 100
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # 9. Plot test data and decision boundary
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=C, cmap=plt.cm.Paired, edgecolors='k')
    plt.title('Test Data with Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Create mesh grid for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Normalize mesh data
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    mesh_data_normalized = (mesh_data - mean) / std
    
    # Get predictions for mesh
    with torch.no_grad():
        Z = net(torch.FloatTensor(mesh_data_normalized)).numpy().reshape(xx.shape)
    
    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='-', linewidths=2)
    
    # Save plot
    plt.savefig(f'decision_boundary_{fileName.split(".")[0]}.pdf')
    plt.close()
    
    return accuracy

task = 'test'  #'train'

if task == 'train':
    fname = "trainSet2.txt"   #input('Enter train file name:')
    TrainMyNet(fname)
    print('done training')
    
    
if task == 'test':
    fname =  "testSet2.txt" #input('Enter test file name:')
    TestMyNet(fname)
    print('done training')    