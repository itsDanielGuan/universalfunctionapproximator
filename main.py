import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
# config for device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Initialise the correct answers
x = np.linspace(-50,50,200)
print(x)
y = np.log(x)
y[np.isnan(y)] = 0
print(y)

# arbitrary hyperparamters
numNeurons = 100
learning_rate = 5e-3 #0.005

# model architecture 1->numNeurons->numNeurons->1
model = nn.Sequential(
    nn.Linear(1,numNeurons),
    nn.ReLU(),
    nn.Linear(numNeurons,numNeurons),
    nn.ReLU(),
    nn.Linear(numNeurons,1)
).to(device)

optimiser = optim.Adam(model.parameters(),lr=learning_rate)

criterion = nn.MSELoss()


# Data ins and outs
inputs = torch.tensor(x).view(-1,1).to(device)
labels = torch.tensor(y).view(-1,1).to(device)
if not os.path.exists("model_parameters.pth"):
    print("Untrained model. Starting Training Cycle.")
    epochs = 20000
    for epoch in range(epochs):
        optimiser.zero_grad()
        
        outputs = model(inputs.float())
        loss = criterion(outputs,labels.float())
        loss.backward()
        if(epoch%100==0):
            plt.scatter(x,y, label="Original")
            plt.scatter(x,outputs.detach().cpu().numpy(),label="Learnt")
            plt.xlabel("Input")
            plt.ylabel("Output")
            plt.legend()
            plt.pause(0.01)
            plt.clf()
            print(f"Epoch:{epoch} Loss:{loss.item()}")
        optimiser.step()
    torch.save(model.state_dict(), 'model_parameters.pth')
else:
    print("Trained weights detected. Using trained model.")
model.load_state_dict(torch.load('model_parameters.pth'))
model.eval()
with torch.no_grad():
    test_inputs = torch.tensor(x).view(-1,1).float().to(device)
    new_output = model.forward(test_inputs).cpu().numpy()

plt.scatter(x,y, label="Original")
plt.scatter(x,new_output,label="Learnt")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.show()
    



