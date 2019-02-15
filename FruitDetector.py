import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = 5
num_classes = 100
batch_size = 100
learning_rate = 0.001


TRAIN_DATA_PATH = "fruits-360/Training/"
TEST_DATA_PATH = "fruits-360/Test/"

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(32),
    #transforms.Grayscale(1),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5] )
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_loader  = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4) 

#Print classes loaded
#print(train_data.class_to_idx)

def imshow(img):
    x=np.transpose(img,(1,2,0))
    plt.imshow(x)
    plt.show()

def findClassNameFromID(dict, id):
    for key,value in dict.items():
        if str(value) == id:
            return key
    return None



# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(8*8*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval() 
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on test images: {} %'.format(100 * correct / total))

    dataiter = iter(test_loader)
    images,labels = dataiter.next()
    listOfFruit=[]
    for j in range(100):
        listOfFruit.append(str(labels[j]).replace("tensor(","").replace(")",""))
    print(' '.join('%5s\t' % findClassNameFromID(train_data.class_to_idx,listOfFruit[j]) for j in range(len(listOfFruit))))

    # show images
    imshow(torchvision.utils.make_grid(images))

torch.save(model.state_dict(), 'model.ckpt')