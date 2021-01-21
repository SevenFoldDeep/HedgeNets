import torch.nn as nn
from ..utils.blocks import normConv, AtrousBlock    
        
class Cloudnet(nn.Module):
    def __init__(self, 
                 res, # width or height of the image where width==height
                 n_classes,
                 n_channels, # number of bands of original image input
                 out_channels, # list
                 dilations, # list
                 kernel, # int
                 padding, # int
                 n_blocks, # int
                 bias = False
                ):
        
        assert len(dilations) == len(out_channels), "'dilations' must be same length as 'out_channels'"
        super(Cloudnet, self).__init__()
        blocks = []
        for i in range(n_blocks):
            if i == 0:
                blocks.append(AtrousBlock(n_channels,
                                        out_channels, 
                                        kernel, 
                                        dilations, 
                                        padding,
                                        i,
                                        bias
                                        ))
            else:
                blocks.append(AtrousBlock(
                                        # plus 1 because we have the 
                                        # normalization layer and then the 
                                        # dilation layers so dilation + norm
                                        out_channels[0]*(len(dilations)+1), 
                                        out_channels, 
                                        kernel, 
                                        dilations, 
                                        padding,
                                        i,
                                        bias
                                        ))
        self.n_class = n_classes
        self.blocks = nn.Sequential(*blocks)
        self.normal = normConv((len(out_channels) +1)*4, n_classes, padding = 0, bias = bias)
        #self.pred = Logits(out_channels[-1], n_classes, bias = bias)

        
    def forward(self, x):
        x = self.blocks(x)
        x = self.normal(x)
        #x = self.pred(x)

        #x = F.softmax(x)
        
        return x
