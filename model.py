#torch: The main PyTorch library.
#torch.nn: PyTorch's neural network module, which includes the definition of neural network layers.
import torch
import torch.nn as nn

#Defines a new class named NeuralNet that inherits from nn.Module.
class NeuralNet(nn.Module):
    #Initializes the neural network architecture.
    def __init__(self, input_size, hidden_size, num_classes):
        #Calls the constructor of the parent class (nn.Module).
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out


#In summary, this code defines a simple neural network class with three linear layers and ReLU activation between them.
#The forward method describes how input data moves through these layers to produce the final output.