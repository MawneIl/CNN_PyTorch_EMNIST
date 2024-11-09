import os
import pickle

import torch
from torch import nn
from torchvision.transforms import Normalize


class CNN(nn.Module):
    
    def __init__(self, n_classes):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten())
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=1024)        
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=n_classes)


    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = out.reshape(out.size(0), -1)
        out = self.drop(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out



class Model():

    def __init__(self, n_classes = 47, checkpoint = 'CNN_ver2.pth', ):
        # variables
        self.n_classes = n_classes
        self.checkpoint = checkpoint
        self.path_mapping = os.path.join('myapp', 'mapping.pkl')
        self.state_dict_path = os.path.join('checkpoints', self.checkpoint)

        # load model
        self.model = CNN(self.n_classes)
        self.model.load_state_dict(torch.load(self.state_dict_path))

        with open(self.path_mapping, 'rb') as file:
            self.mapping = pickle.load(file)

    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        predict : str
            Символ-предсказание 
        '''
        while len(x.shape) < 4:
            x = x.unsqueeze(0)

        transform = Normalize([0.5], [0.5])
        x = transform(x.float())

        self.model.eval()
        output = self.model(x).argmax(dim=1)
        predict = self.mapping[int(output)]
        
        return predict