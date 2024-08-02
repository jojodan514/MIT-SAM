import torch
import torch.nn as nn
import torch.nn.functional as F

class Edge_reconstruct(nn.Module):

    def __init__(self, image_size):

        super(Edge_reconstruct, self).__init__()

        self.image_size = image_size

        self.edge_encoder = double_conv2d_bn(in_channels=1,
                                             out_channels=64,
                                             kernel_size=3,
                                             strides=2,
                                             padding=1)

        self.edge_decoder = deconv2d_bn(in_channels=64,
                                        out_channels=1,
                                        kernel_size=2,
                                        strides=2)

    def forward(self, x):
        x = self.edge_encoder(x)
        x = self.edge_decoder(x)

        return x
    
class double_conv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(double_conv2d_bn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,
                               kernel_size=kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,
                              kernel_size = kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
    
class conv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(conv2d_bn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,
                               kernel_size=kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out



class deconv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(deconv2d_bn,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels,out_channels,
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))        
        return out





class double_deconv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(double_deconv2d_bn,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels,out_channels,
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.conv2 = nn.ConvTranspose2d(out_channels,out_channels,
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        
        return out


class Multi_scale_layer(nn.Module):
    def __init__(self,channel=[3,16,32,64,128],kernel_size=2,strides=2):
        super(Multi_scale_layer,self).__init__()
        self.conv1 = nn.ConvTranspose2d(channel[0],channel[1],
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.conv2 = nn.ConvTranspose2d(channel[1],channel[2],
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.conv3 = nn.ConvTranspose2d(channel[2],channel[3],
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.conv4 = nn.ConvTranspose2d(channel[3],channel[4],
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.conv_bn1 = nn.BatchNorm2d(channel[1])
        self.conv_bn2 = nn.BatchNorm2d(channel[2])
        self.conv_bn3 = nn.BatchNorm2d(channel[3])
        self.conv_bn4 = nn.BatchNorm2d(channel[4])

        self.de_conv1 = nn.ConvTranspose2d(channel[4],channel[3],
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.de_conv2 = nn.ConvTranspose2d(channel[3],channel[2],
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.de_conv3 = nn.ConvTranspose2d(channel[2],channel[1],
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.de_conv4 = nn.ConvTranspose2d(channel[1],channel[0],
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        
        self.deconv_bn1 = nn.BatchNorm2d(channel[3])
        self.deconv_bn2 = nn.BatchNorm2d(channel[2])
        self.deconv_bn3 = nn.BatchNorm2d(channel[1])
        self.deconv_bn4 = nn.BatchNorm2d(channel[0])

    def forward(self,x):

        conv_list = []
        deconv_list = []
        out = F.relu(self.conv_bn1(self.conv1(x)))
        conv_list.append(out)
        out = F.relu(self.conv_bn2(self.conv2(out)))
        conv_list.append(out)
        out = F.relu(self.conv_bn3(self.conv3(out)))
        conv_list.append(out)
        out = F.relu(self.conv_bn4(self.conv4(out)))

        out = F.relu(self.deconv_bn1(self.de_conv1(out)))
        out += conv_list[2]
        out = F.relu(self.deconv_bn2(self.de_conv2(out)))
        out += conv_list[1]
        out = F.relu(self.deconv_bn3(self.de_conv3(out)))
        out += conv_list[0]
        out = F.relu(self.deconv_bn4(self.de_conv4(out)))
        out += x
        
        return out



    

if __name__ == '__main__':
    x = torch.rand((8,3,512,512))
    multi_layer = Multi_scale_layer()
    y = multi_layer(x)
    print(y.shape)