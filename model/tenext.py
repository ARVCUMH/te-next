import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
from model.resnet_custom import ResNetBase
from utils import SparseTensorLayerNorm , SparseTensorLinear, assign_feats
from timm.models.layers import trunc_normal_, DropPath
# from timm.models.layers import DropPath, trunc_normal_

class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        
        self.conv0p1s1 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = SparseTensorLayerNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = SparseTensorLayerNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],self.LAYERS[0],dpr=self.dpr[sum(self.LAYERS[:0]):sum(self.LAYERS[:1])])

        self.conv2p2s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = SparseTensorLayerNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],self.LAYERS[1],dpr=self.dpr[sum(self.LAYERS[:1]):sum(self.LAYERS[:2])])
        # self.fpn2=SparseTensorLinear(self.PLANES[1], self.PLANES[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn3 = SparseTensorLayerNorm(self.inplanes)
        
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],self.LAYERS[2], dpr=self.dpr[sum(self.LAYERS[:2]):sum(self.LAYERS[:3])])
        # self.fpn3=SparseTensorLinear(self.PLANES[2], self.PLANES[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = SparseTensorLayerNorm(self.inplanes)

        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],self.LAYERS[3], dpr=self.dpr[sum(self.LAYERS[:3]):sum(self.LAYERS[:4])])
      

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = SparseTensorLayerNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],self.LAYERS[4], dpr=self.dpr[sum(self.LAYERS[:4]):sum(self.LAYERS[:5])])


        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = SparseTensorLayerNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],self.LAYERS[5], dpr=self.dpr[sum(self.LAYERS[:5]):sum(self.LAYERS[:6])])
        
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = SparseTensorLayerNorm(self.PLANES[6])

        # self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.inplanes = self.PLANES[6]
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],self.LAYERS[6], dpr=self.dpr[sum(self.LAYERS[:6]):sum(self.LAYERS[:7])])
        
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = SparseTensorLayerNorm(self.PLANES[7])

        # self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.inplanes = self.PLANES[7]
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7], dpr=self.dpr[sum(self.LAYERS[:7]):sum(self.LAYERS[:8])])
        
        self.final = ME.MinkowskiLinear(self.PLANES[7], out_channels)
        # self.final = SparseTensorLinear(self.PLANES[7], out_channels)
        self.relu = ME.MinkowskiReLU()
        #self.relu=ME.MinkowskiELU()
        # self.relu=ME.MinkowskiGELU()
        
        self.softmax= ME.MinkowskiSigmoid()


    def forward(self, x):

        out = self.conv0p1s1(x)

        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)

        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)

        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)


        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)

        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8

        out = self.convtr4p16s2(out)

        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)
    
        # # # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # # tensor_stride=2

        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        out = self.block7(out)
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        out = self.block8(out)
        out=self.final(out)
        return self.softmax(out)


        

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.F.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output=assign_feats(x,x.F.div(keep_prob) * random_tensor)
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class TENext_block(nn.Module):
    expansion=1
    def __init__(self, inplanes, planes, drop_path_rate, stride=1, dilation=1, downsample=None, bn_momentum=0.1, dimension=-1):
        super().__init__()
        d_2 = planes //4 #cambiar la division por multiplicacion 
        self.unary_1 = nn.Sequential(
            SparseTensorLinear(inplanes, d_2, bias=False),
            SparseTensorLayerNorm(d_2),
            # ME.MinkowskiReLU(), #probar a quitar las relus y meter una gelu
        )
        self.unary_2 = nn.Sequential(
            SparseTensorLinear(d_2, planes, bias=False),
            SparseTensorLayerNorm(planes),
            # ME.MinkowskiReLU(),
        )
        self.spconv = ME.MinkowskiConvolution( # El original es el 5. La siguiente prueba vuelvas a poner el 5
            d_2,
            d_2,
            kernel_size=7,
            stride=1,
            dilation=1,
            bias=False,
            dimension=3,
        )
        if inplanes != planes:
            self.shortcut_op = nn.Sequential(
                SparseTensorLinear(inplanes, planes, bias=False),
                SparseTensorLayerNorm(planes),
            )
        else:
            self.shortcut_op = nn.Identity()

        self.drop_path = DropPath(drop_path_rate[0]) if drop_path_rate[0] > 0.0 else nn.Identity()
        self.gelu = ME.MinkowskiGELU()

    def forward(self, x):
        
        shortcut_f = x.F.clone()
        shortcut=assign_feats(x, shortcut_f)
        x = self.unary_1(x)
        x = self.spconv(x)
        # x = self.gelu(x)
        x = self.unary_2(x) 
        x = self.drop_path(x)
        x=self.gelu(x) 

        shortcut = self.shortcut_op(shortcut) 
        x += shortcut 
        return x



   
class TENext(MinkUNetBase):
    BLOCK = TENext_block
    LAYERS = [1, 1, 1, 1, 1, 1, 1, 1]
    DROP_RATE=0.3
    dpr = [x.item() for x in torch.linspace(0, DROP_RATE, sum(LAYERS))] #ir haciendo dpr[0], dpr[1], etc..., es igual a dpr[sum(depths[:i]):sum(depths[:i+1])]
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)




