import torch.nn as nn
import torch.nn.functional as F 
import torch
class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        # print(f'after convnet' + str(output.size()))
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_encoding(self, x):
        return self.forward(x)
    
    
class TripletNet(nn.Module):
    def __init__(self, encoder):
        super(TripletNet, self).__init__()
        self.encoder_net = encoder

    def forward(self, x1, x2, x3):
        output1 = self.encoder_net(x1)
        output2 = self.encoder_net(x2)
        output3 = self.encoder_net(x3)
        return output1, output2, output3

    def get_encoding(self, x):
        return self.encoder_net(x)