import torch
import torch.nn as nn
from ptt.models.model import Model
from gd.models.segmentation.unet import UNet


class MMUNet(Model):

    def __init__(self, model_g, model_h, input_shape=(1, 64, 64), nr_class=1, nr_input_layers=10):

        # Double input channels
        input_shape = list(input_shape)
        input_shape[0]*=2
        input_shape = tuple(input_shape)

        # Initialize super model
        super().__init__(input_shape=input_shape, output_shape=input_shape)
        nr_input_channels = min(input_shape)

        # Initialize submodels
        self.g = model_g
        self.h = model_h
        self.f_layers = UNet(input_shape=input_shape, nr_class=nr_class, nr_input_layers=nr_input_layers)

    def predict(self, x):
        inputs = self.preprocess_input(x)
        outputs = self.forward(inputs)
        outputs = torch.sigmoid(outputs)
        return outputs
        
    def forward(self, x):
        #print('forward')
        #print(x.shape)
        x1 = self.g(x)
        #print(x1.shape)
        # x1 = torch.sigmoid(x1)
        x2 = self.h(x)
        #print(x2.shape)
        # x2 = torch.sigmoid(x2)
        x = torch.cat((x1, x2), dim=1)
        #print(x.shape)
        x = self.f_layers(x)
        #print(x.shape)
        return x