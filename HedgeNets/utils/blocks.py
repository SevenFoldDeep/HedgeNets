import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel, padding, bias):
    return nn.Conv2d(in_channels, 
                     out_channels, 
                     kernel_size = kernel, 
                     padding = padding, 
                     bias = bias)

def normConv(in_channels, out_channels, padding, bias):
    return nn.Conv2d(in_channels, 
                     out_channels, 
                     kernel_size = 1, 
                     padding = 0, 
                     bias = bias)

class ConvLayer(nn.Sequential):
    def __init__(self, 
                 n_block, 
                 n_layer, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 dilation,
                 padding, 
                 bias):
        super(ConvLayer, self).__init__()
        
        self.add_module('conv%d_%d' % (n_block, n_layer), 
                        nn.Conv2d(in_channels, 
                                  out_channels, 
                                  kernel_size, 
                                  padding = padding, 
                                  bias = bias)
                       )
        self.add_module('bnorm%d_%d' % (n_block, n_layer), 
                        nn.BatchNorm2d(out_channels)
                        )
        self.add_module('relu%d_%d' % (n_block, n_layer), 
                        nn.ReLU()
                        )

class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel, 
                 block_num,
                 bias,
                 down = True
                 ):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = ConvLayer(block_num,
                               1,
                               in_channels, 
                               out_channels, 
                               kernel_size = 3, 
                               padding = 1, 
                               bias = bias)
        self.conv2 = ConvLayer(block_num,
                               2,
                               out_channels, 
                               out_channels, 
                               kernel_size = 3, 
                               padding = 1, 
                               bias = bias)
        if down:                       
            self.downsample = nn.MaxPool2d(2, 2)
        else:
            self.downsample = None
        
    def forward(self, x):
        if self.downsample is not None:
            x = self.conv1(x)
            residual = self.conv2(x)
            out = self.downsample(residual)
            return residual, out
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x
            
            
class DeconvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel, 
                 block_num,
                 bias
                 ):
        
        super(DeconvBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 
                                           kernel_size=2, stride=2)
        self.conv1 = ConvLayer(block_num,
                               1,
                               in_channels, 
                               out_channels, 
                               kernel, 
                               padding = 1,
                               bias = bias)
        self.conv2 = ConvLayer(block_num,
                               2,
                               out_channels, 
                               out_channels, 
                               kernel, 
                               padding = 1,
                               bias = bias)
        
        
    def forward(self, x, residual):
        x = self.upsample(x)
        x = torch.cat((x, residual), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Logits(nn.Sequential):
    def __init__(self, 
                 in_channels, 
                 n_class
                 ):
      super(Logits, self).__init__()

      self.conv = self.add_module('conv_out', 
                        nn.Conv2d(in_channels, 
                                  n_class, 
                                  kernel_size = 1
                                  )
                       )
                       
                       
class AtrousConv(nn.Sequential):
    def __init__(self, 
                 n_block, 
                 n_layer, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 padding, 
                 dilation,
                 bias):
        super(AtrousConv, self).__init__()
        
        self.add_module('conv%d_%d' % (n_block, n_layer), 
                        nn.Conv2d(in_channels, 
                                  out_channels, 
                                  kernel_size, 
                                  padding = padding, 
                                  dilation = dilation,
                                  bias = bias)
                       )
        self.add_module('bnorm%d_%d' % (n_block, n_layer), 
                        nn.BatchNorm2d(out_channels)
                        )
        self.add_module('relu%d_%d' % (n_block, n_layer), 
                        nn.ReLU()
                        )
        
class AtrousBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel, 
                 dilations, 
                 padding,
                 block_num,
                 bias
                 ):
        
        super(AtrousBlock, self).__init__()
        
        # normalize the number of incoming layers
        self.norm = normConv(in_channels, out_channels, padding, bias) 
        
        #atrousconvs = []
        #for dilate in dilations:
        #    atrousconvs.append(AtrousConv(block_num,
        #                       1,
        #                       4, 
        #                       out_channels[0], 
        #                       kernel, 
        #                       padding = padding + dilations[0] -1, 
        #                       dilation = dilate,
        #                       bias = bias)
        #                       )
        
        
        #self.aspp = nn.Sequential(*atrousconvs)
        self.aspp1 = AtrousConv(block_num,
                               1,
                               out_channels, 
                               out_channels, 
                               kernel, 
                               padding = padding + dilations[0] -1, 
                               dilation = dilations[0],
                               bias = bias)
        self.aspp2 = AtrousConv(block_num,
                               2,
                               out_channels, 
                               out_channels, 
                               kernel, 
                               padding = padding + dilations[1]-1, 
                               dilation = dilations[1],
                               bias = bias)
        self.aspp3 = AtrousConv(block_num,
                               3,
                               out_channels, 
                               out_channels, 
                               kernel, 
                               padding = padding + dilations[2]-1, 
                               dilation = dilations[2],
                               bias = bias)
        self.aspp4 = AtrousConv(block_num,
                               4,
                               out_channels, 
                               out_channels, 
                               kernel, 
                               padding = padding + dilations[3]-1, 
                               dilation = dilations[3],
                               bias = bias)
        self.aspp5 = AtrousConv(block_num,
                               4,
                               out_channels, 
                               out_channels, 
                               kernel, 
                               padding = padding + dilations[4]-1, 
                               dilation = dilations[4],
                               bias = bias)
        self.aspp6 = AtrousConv(block_num,
                               4,
                               out_channels, 
                               out_channels, 
                               kernel, 
                               padding = padding + dilations[5]-1, 
                               dilation = dilations[5],
                               bias = bias)
        self.aspp7 = AtrousConv(block_num,
                               4,
                               out_channels, 
                               out_channels, 
                               kernel, 
                               padding = padding + dilations[6]-1, 
                               dilation = dilations[6],
                               bias = bias)
    
        self.bn_out = nn.BatchNorm2d(out_channels* (len(dilations) + 1)) 
        self.relu_out = nn.ReLU()
        
    def forward(self, x):
        x1 = self.norm(x) #residual
        x2 = self.aspp1(x1)
        x3 = self.aspp2(x1)
        x4 = self.aspp3(x1)
        x5 = self.aspp4(x1)
        x6 = self.aspp5(x1)
        x7 = self.aspp6(x1)
        x8 = self.aspp7(x1)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 
                      dim=1) # CORRECT CONCAT DIM?
        x = self.bn_out(x)
        x = self.relu_out(x)
        return x