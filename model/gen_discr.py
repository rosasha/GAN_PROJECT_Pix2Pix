#!/usr/bin/env python
# coding: utf-8

# In[1]:



import torch
import torch.nn as nn


# In[2]:


class CBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, act=True, batch_norm=True):
        super(CBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = act
        self.lrelu = nn.LeakyReLU(0.2)
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out


# In[3]:


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super(DBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU(True)
        self.batch_norm = batch_norm
        self.dropout = dropout
        
    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out


# In[17]:


class Generator(torch.nn.Module):
    def __init__(self, in_channels=3, features=64,):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = CBlock(in_channels, features, batch_norm=False)
        self.conv2 = CBlock(features, features * 2)
        self.conv3 = CBlock(features * 2, features * 4)
        self.conv4 = CBlock(features * 4, features * 8)
        self.conv5 = CBlock(features * 8, features * 8)
        self.conv6 = CBlock(features * 8, features * 8)
        self.conv7 = CBlock(features * 8, features * 8)
        self.bottleneck = CBlock(features * 8, features * 8, batch_norm=False)
        
        # Decoder
        self.deconv1 = DBlock(features * 8, features * 8, dropout=True)
        self.deconv2 = DBlock(features * 8 * 2, features * 8, dropout=True)
        self.deconv3 = DBlock(features * 8 * 2, features * 8, dropout=True)
        self.deconv4 = DBlock(features * 8 * 2, features * 8)
        self.deconv5 = DBlock(features * 8 * 2, features * 4)
        self.deconv6 = DBlock(features * 4 * 2, features * 2)
        self.deconv7 = DBlock(features * 2 * 2, features)
        self.final_up = DBlock(features * 2, in_channels, batch_norm=False)
        
    def forward(self, x):
        # Encoder
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)
        d7 = self.conv7(d6)
        bottleneck = self.bottleneck(d7)
        
        # Decoder
        up1 = self.deconv1(bottleneck)
        up2 = self.deconv2(torch.cat([up1, d7], 1))
        up3 = self.deconv3(torch.cat([up2, d6], 1))
        up4 = self.deconv4(torch.cat([up3, d5], 1))
        up5 = self.deconv5(torch.cat([up4, d4], 1))
        up6 = self.deconv6(torch.cat([up5, d3], 1))
        up7 = self.deconv7(torch.cat([up6, d2], 1))
        out = self.final_up(torch.cat([up7, d1], 1))
        out = nn.Tanh()(out)
        return out


# In[24]:


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,  features=64):
        super(Discriminator, self).__init__()

        self.conv1 = CBlock(in_channels*2, features, act=False, batch_norm=False)
        self.conv2 = CBlock(features, features * 2)
        self.conv3 = CBlock(features * 2, features * 4)
        self.conv4 = CBlock(features * 4, features * 8, stride=1)
        self.conv5 = CBlock(features * 8, out_channels, stride=1, batch_norm=False)
        
    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = nn.Sigmoid()(x)
        return out


# In[30]:


def test1():
    x = torch.randn((1, 3, 256, 256))
    model1 = Generator(in_channels=3, features=64)
    preds1 = model1(x)
    print(model1)
    print(preds1.shape)
    
    
def test2():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model2 = Discriminator(in_channels=3, out_channels=1, features=64)
    preds2 = model2(x, y)
    print(model2)
    print(preds2.shape)


if __name__ == "__main__":
    test1()
    test2()

