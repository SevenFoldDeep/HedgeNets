import torch.nn as nn
from ..utils.blocks import normConv, AtrousBlock    
        
class Cloudnet(nn.Module):
    def __init__(self, 
                 res, # width or height of the image where width==height
                 n_classes,
                 n_channels, # number of bands of original image input
                 out_channels, # int
                 dilations, # list
                 kernel, # int
                 padding, # int
                 n_blocks, # int
                 bias = False
                ):
        
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
                                        )
                              )
            else:
                blocks.append(AtrousBlock(
                                        # plus 1 because we have the 
                                        # normalization layer and then the 
                                        # dilation layers so dilation + norm
                                        out_channels*(len(dilations)+1), 
                                        out_channels, 
                                        kernel, 
                                        dilations, 
                                        padding,
                                        i,
                                        bias
                                        )
                            )
        self.n_class = n_classes
        self.blocks = nn.Sequential(*blocks)
        self.normal = normConv((len(dilations) + 1)*out_channels, 
                               n_classes, 
                               padding = 0, 
                               bias = bias)

        #self.pred = Logits(out_channels, n_classes, bias = bias)

        
    def forward(self, x):
        x = self.blocks(x)
        x = self.normal(x)
        #x = self.pred(x)

        #x = F.softmax(x)
        
        return x

