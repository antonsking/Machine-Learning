'''
___  ___      _ _   _ _                        ______                       _                   
|  \/  |     | | | (_) |                       | ___ \                     | |                  
| .  . |_   _| | |_ _| | __ _ _   _  ___ _ __  | |_/ /__ _ __ ___ ___ _ __ | |_ _ __ ___  _ __  
| |\/| | | | | | __| | |/ _` | | | |/ _ \ '__| |  __/ _ \ '__/ __/ _ \ '_ \| __| '__/ _ \| '_ \ 
| |  | | |_| | | |_| | | (_| | |_| |  __/ |    | | |  __/ | | (_|  __/ |_) | |_| | | (_) | | | |
\_|  |_/\__,_|_|\__|_|_|\__,_|\__, |\___|_|    \_|  \___|_|  \___\___| .__/ \__|_|  \___/|_| |_|
                               __/ |                                 | |                        
                              |___/                                  |_|                       
'''

import numpy as np
import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_epochs, learning_rate=0.1):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(input_size,hidden_size,bias=True),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size,output_size))
       
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.input_size = input_size
        

    def forward(self, x):
        return self.model(x)
    

    def fit(self,dataloader,criterion,optimizer):
        train_loss = []
        train_acc = []

        for i in range(self.max_epochs):
            total_loss = 0
            for j,(images,labels) in enumerate(dataloader):
                
                images = images.view(images.shape[0], -1)
                optimizer.zero_grad()
                
                # Forward pass (consider the recommmended functions in HW4 writeup)
                output = self.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                # Track the accuracy
                total_loss += loss.item()
                result = torch.max(output.data,1)
                tot = 0
                for k,pred in enumerate(result[1]):
                    if pred.item() == labels[k].item():
                        tot+=1
                
            train_loss.append(total_loss/(j+1))
            train_acc.append(tot/len(labels))

        return train_loss,train_acc


    def predict(self,dataloader,criterion):
        test_loss = []
        test_acc = []
        
        with torch.no_grad():
            for j,(images,labels) in enumerate(dataloader):
                tot = []
                # compute output and loss
                images = images.view(images.shape[0], -1)
                output = self.forward(images)
                loss = criterion(output, labels)

                # measure accuracy and record loss
                result = torch.max(output.data,1)
                for k,pred in enumerate(result[1]):
                    if pred.item() == labels[k].item():
                        tot.append(1)
                        
                test_loss.append(loss.item())
                test_acc.append(sum(tot)/len(labels))
        
        test_loss = sum(test_loss) / len(test_loss)
        test_acc = sum(test_acc) / len(test_acc)
        
        return test_loss, test_acc