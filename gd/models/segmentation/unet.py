import torch
import torch.nn as nn
from ptt.models.model import Model

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )    

class UNet(Model):

    def __init__(self, input_shape=(1, 64, 64), nr_class=1, nr_input_layers=10):

        super().__init__(input_shape=input_shape, output_shape=input_shape)
        nr_input_channels = min(input_shape)

        self.dconv_down1 = double_conv(nr_input_channels, nr_input_layers)
        self.dconv_down2 = double_conv(nr_input_layers, nr_input_layers*2)
        self.dconv_down3 = double_conv(nr_input_layers*2, nr_input_layers*4)
        self.dconv_down4 = double_conv(nr_input_layers*4, nr_input_layers*8)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(nr_input_layers*4 + nr_input_layers*8, nr_input_layers*4)
        self.dconv_up2 = double_conv(nr_input_layers*2 + nr_input_layers*4, nr_input_layers*2)
        self.dconv_up1 = double_conv(nr_input_layers*2 + nr_input_layers, nr_input_layers)
        
        self.conv_last = nn.Conv2d(nr_input_layers, nr_class, 1)


    def get_layer(self, key):
            return getattr(self, key)

    def predict(self, x):
        inputs = self.preprocess_input(x)
        outputs = self.forward(inputs)
        outputs = torch.sigmoid(outputs)
        return outputs
        
    def forward(self, x):
        conv1 = self.get_layer('dconv_down1')(x)
        #conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.get_layer('dconv_down2')(x)
        #conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.get_layer('dconv_down3')(x)
        #conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.get_layer('dconv_down4')(x)
        #x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.get_layer('dconv_up3')(x)
        #x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.get_layer('dconv_up2')(x)
        #x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.get_layer('dconv_up1')(x)
        #x = self.dconv_up1(x)
        
        out = self.get_layer('conv_last')(x)
        #out = self.conv_last(x)
        
        return out