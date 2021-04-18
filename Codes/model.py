import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)




def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    
    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")
    
    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length)) 
        for start, length in zip(splits, split_sizes))


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)                    
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()            


def init_weights_constant(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.weight, 0.3)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.constant_(m.weight, 0.3)
        if m.bias is not None:
            m.bias.data.fill_(0.01)                    
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def convbn(in_planes, out_planes, kernel_size, stride, padding = 1, statsFlag = True):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding = padding, bias=False),
                         nn.BatchNorm2d(out_planes, track_running_stats = statsFlag))



def conv(in_planes, out_planes, kernel_size=3,stride=1):
    return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
		nn.BatchNorm2d(out_planes),
		nn.ReLU(inplace=True)
	)

def deconv(in_planes, out_planes):
    return nn.Sequential(
		nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
		nn.BatchNorm2d(out_planes),
		nn.ReLU(inplace=True)
	)


def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True, statsFlag = True):
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels, track_running_stats = statsFlag))
    if relu:
        #layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.ReLU(inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

class DepthEncoder(nn.Module):
    def __init__(self, in_layers = 2, layers = 32, filter_size = 3):
        super(DepthEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)
        dl_padding = (filter_size - 1)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),

                                  )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),

                                  )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x=None):


        ### input

        x0 = self.init(input)
        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=True)

        ### hourglass with short cuts connections between encoder and decoder
        x1 = self.enc1(x0) #1/2 input size
        x2 = self.enc2(x1) # 1/4 input size

        return x0, x1, x2


class RGBEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(RGBEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc3 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc4 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x=None):

        ### input

        x0 = self.init(input)
        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=True)

        ### hourglass with short cuts connections between encoder and decoder
        x1 = self.enc1(x0)  # 1/2 input size
        x2 = self.enc2(x1)  # 1/4 input size
        x3 = self.enc3(x2)  # 1/8 input size
        x4 = self.enc4(x3)  # 1/16 input size

        return x0, x1, x2, x3, x4

class DepthDecoder(nn.Module):
    def __init__(self, layers, filter_size):
        super(DepthDecoder, self).__init__()
        padding = int((filter_size-1)/2)

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers//2, layers//2, filter_size, stride=2, padding=padding, output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers//2, layers//2, filter_size, stride=1, padding=padding),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers//2, layers//2, filter_size, stride=2, padding=padding, output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers//2, layers//2, filter_size, stride=1, padding=padding),
                                  )

        self.prdct = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(layers//2, layers//2, filter_size, stride=1, padding=padding),
                                     nn.ReLU(),
                                     nn.Conv2d(layers//2, 1, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_dx, pre_cx):

        x2 = pre_dx[2] + pre_cx[2]#torch.cat((pre_dx[2], pre_cx[2]), 1)
        x1 = pre_dx[1] + pre_cx[1]#torch.cat((pre_dx[1], pre_cx[1]), 1)
        x0 = pre_dx[0] + pre_cx[0]


        x3 = self.dec2(x2) # 1/2 input size
        x4 = self.dec1(x1+x3) #1/1 input size

        ### prediction
        output_d = self.prdct(x4+x0)

        return x4, output_d

class DepthDecoder_3chandep(nn.Module):
    def __init__(self, layers, filter_size, out_chan = 3):
        super(DepthDecoder_3chandep, self).__init__()
        padding = int((filter_size-1)/2)

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers//2, layers//2, filter_size, stride=2, padding=padding, output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers//2, layers//2, filter_size, stride=1, padding=padding),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers//2, layers//2, filter_size, stride=2, padding=padding, output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers//2, layers//2, filter_size, stride=1, padding=padding),
                                  )

        self.prdct = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(layers//2, layers//2, filter_size, stride=1, padding=padding),
                                     nn.ReLU(),
                                     nn.Conv2d(layers//2, out_chan, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_dx, pre_cx):

        x2 = pre_dx[2] + pre_cx[2]#torch.cat((pre_dx[2], pre_cx[2]), 1)
        x1 = pre_dx[1] + pre_cx[1]#torch.cat((pre_dx[1], pre_cx[1]), 1)
        x0 = pre_dx[0] + pre_cx[0]


        x3 = self.dec2(x2) # 1/2 input size
        x4 = self.dec1(x1+x3) #1/1 input size

        ### prediction
        output_d = self.prdct(x4+x0)

        return x4, output_d


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers+cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)


    def forward(self, input_d, input_rgb, C):

        enc_c = self.rgb_encoder(input_rgb)

        ## for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[1], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[1] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[1] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(dcd_d14[1], scale_factor=4, mode='bilinear', align_corners=True)

        return  output_d11, output_d12,output_d14,

class final_network(nn.Module):
    def __init__(self):
        super(final_network, self).__init__()
        self.module = network()

    def forward(self, input_d, input_rgb, C):        

        output_d11, output_d12, output_d14 = self.module(input_d, input_rgb, C)

        return output_d11, output_d12, output_d14


class MultiRes_network_diffspatialsizes(nn.Module):
    def __init__(self):
        super(MultiRes_network_diffspatialsizes, self).__init__()

        denc_layers = 64
        cenc_layers = 64
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder_3chandep(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder_3chandep(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder_3chandep(ddcd_layers, 3)

    def forward(self, input_d, input_rgb):

        enc_c = self.rgb_encoder(input_rgb)
        
        ## for the 1/4 res
        input_d14 = F.max_pool2d(input_d, 4, 4)
        input_d12 = F.max_pool2d(input_d, 2, 2) 
        
        #input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        #input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)

        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])
        output_d14 = dcd_d14[1]  #predicted 3 chan dep in 1/4 res        
        
        
        dcd_d14_splitchannels = torch.split(dcd_d14[1], 1, 1)                
        
        sig_d14 = torch.sigmoid(dcd_d14_splitchannels[2])
        pred_depfus_d14 = sig_d14*dcd_d14_splitchannels[0] + (1 - sig_d14)*dcd_d14_splitchannels[1]  #predicted fused dep in 1/4 res
        
        output_d14_2chan_interp = torch.cat((dcd_d14_splitchannels[0], dcd_d14_splitchannels[1]), 1)  #predicted 2 chan dep in 1/4 res
        ## for the 1/2 res
        
        predict_d12 = F.interpolate(pred_depfus_d14, scale_factor = 2, mode='bilinear', align_corners=True)  #fused dep scaled up to half res.
        output_d14_2chan_interp = F.interpolate(output_d14_2chan_interp, scale_factor = 2, mode = 'bilinear', align_corners = True) #predicted 2 chan dep scaled to half res
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])
        
        ## for the 1/1 res
        dcd_d12_splitchannels = torch.split(dcd_d12[1], 1, 1)        
        sig_d12 = torch.sigmoid(dcd_d12_splitchannels[2])
        pred_depfus_d12 = sig_d12*dcd_d12_splitchannels[0] + (1 - sig_d12)*dcd_d12_splitchannels[1]
        output_d12_2chan_interp = torch.cat((dcd_d12_splitchannels[0], dcd_d12_splitchannels[1]), 1)        
        output_d12_2chan_interp = output_d12_2chan_interp + output_d14_2chan_interp
        output_d12 = torch.cat((output_d12_2chan_interp, dcd_d12_splitchannels[2]), 1)

        predict_d11 = F.interpolate(pred_depfus_d12 + predict_d12, scale_factor = 2, mode='bilinear', align_corners=True)
        output_d12_2chan_interp = F.interpolate(output_d12_2chan_interp, scale_factor = 2, mode = 'bilinear', align_corners = True)

        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])
        
        dcd_d11_splitchannels = torch.split(dcd_d11[1], 1, 1)
        output_d11 = torch.cat((dcd_d11_splitchannels[0], dcd_d11_splitchannels[1]), 1)
        output_d11 = output_d11 + output_d12_2chan_interp
        output_d11 = torch.cat((output_d11, dcd_d11_splitchannels[2]), 1)        

        return output_d11, output_d12, output_d14


class MultiRes_network_avgpool_diffspatialsizes(nn.Module):
    def __init__(self):
        super(MultiRes_network_avgpool_diffspatialsizes, self).__init__()

        denc_layers = 64
        cenc_layers = 64
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder_3chandep(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder_3chandep(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder_3chandep(ddcd_layers, 3)

    def forward(self, input_d, input_rgb):
        C = (input_d > 0).float()
        enc_c = self.rgb_encoder(input_rgb)
        
        ## for the 1/4 res
        
        #input_d14 = F.max_pool2d(input_d, 4, 4)
        #input_d12 = F.max_pool2d(input_d, 2, 2) 

        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)

        input_d14[input_d14 > input_d.max()] = 0
        input_d12[input_d12 > input_d.max()] = 0

        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])
        output_d14 = dcd_d14[1]  #predicted 3 chan dep in 1/4 res        
        
        
        dcd_d14_splitchannels = torch.split(dcd_d14[1], 1, 1)                
        
        sig_d14 = torch.sigmoid(dcd_d14_splitchannels[2])
        pred_depfus_d14 = sig_d14*dcd_d14_splitchannels[0] + (1 - sig_d14)*dcd_d14_splitchannels[1]  #predicted fused dep in 1/4 res
        
        output_d14_2chan_interp = torch.cat((dcd_d14_splitchannels[0], dcd_d14_splitchannels[1]), 1)  #predicted 2 chan dep in 1/4 res
        ## for the 1/2 res
        
        predict_d12 = F.interpolate(pred_depfus_d14, scale_factor = 2, mode='bilinear', align_corners=True)  #fused dep scaled up to half res.
        output_d14_2chan_interp = F.interpolate(output_d14_2chan_interp, scale_factor = 2, mode = 'bilinear', align_corners = True) #predicted 2 chan dep scaled to half res
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])
        
        ## for the 1/1 res
        dcd_d12_splitchannels = torch.split(dcd_d12[1], 1, 1)        
        sig_d12 = torch.sigmoid(dcd_d12_splitchannels[2])
        pred_depfus_d12 = sig_d12*dcd_d12_splitchannels[0] + (1 - sig_d12)*dcd_d12_splitchannels[1]
        output_d12_2chan_interp = torch.cat((dcd_d12_splitchannels[0], dcd_d12_splitchannels[1]), 1)        
        output_d12_2chan_interp = output_d12_2chan_interp + output_d14_2chan_interp
        output_d12 = torch.cat((output_d12_2chan_interp, dcd_d12_splitchannels[2]), 1)

        predict_d11 = F.interpolate(pred_depfus_d12 + predict_d12, scale_factor = 2, mode='bilinear', align_corners=True)
        output_d12_2chan_interp = F.interpolate(output_d12_2chan_interp, scale_factor = 2, mode = 'bilinear', align_corners = True)

        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])
        
        dcd_d11_splitchannels = torch.split(dcd_d11[1], 1, 1)
        output_d11 = torch.cat((dcd_d11_splitchannels[0], dcd_d11_splitchannels[1]), 1)
        output_d11 = output_d11 + output_d12_2chan_interp
        output_d11 = torch.cat((output_d11, dcd_d11_splitchannels[2]), 1)        

        return output_d11, output_d12, output_d14,

class MultiRes_network_samespatialsizes(nn.Module):
    def __init__(self):
        super(MultiRes_network_samespatialsizes, self).__init__()

        denc_layers = 64
        cenc_layers = 64
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder_3chandep(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(3, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder_3chandep(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(3, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder_3chandep(ddcd_layers, 3)

    def forward(self, input_d, input_rgb):

        enc_c = self.rgb_encoder(input_rgb)

        ## for the 1/4 res
        input_d14 = F.max_pool2d(input_d, 4, 4)
        input_d12 = F.max_pool2d(input_d, 2, 2) 
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])        
        
        
        dcd_d14_splitchannels = torch.split(dcd_d14[1], 1, 1)                
        
        sig_d14 = torch.sigmoid(dcd_d14_splitchannels[2])
        output_d14_2chan_interp = torch.cat((dcd_d14_splitchannels[0], dcd_d14_splitchannels[1]), 1)  #predicted 2 chan dep in 1/4 res
        ## for the 1/2 res
        
        predict_d12 = F.interpolate(output_d14_2chan_interp, scale_factor = 2, mode='bilinear', align_corners=True)  #fused dep scaled up to half res.        
        predict_d14_scaled = F.interpolate(output_d14_2chan_interp, scale_factor = 4, mode='bilinear', align_corners=True)  #fused dep scaled up to half res.        
        output_d14_sigmoid_interp = F.interpolate(sig_d14, scale_factor = 4, mode = 'nearest')
        
        output_d14 = torch.cat((predict_d14_scaled, output_d14_sigmoid_interp), 1)        
        
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])
        
        ## for the 1/1 res
        dcd_d12_splitchannels = torch.split(dcd_d12[1], 1, 1)        
        sig_d12 = torch.sigmoid(dcd_d12_splitchannels[2])        
        output_d12_2chan_interp = torch.cat((dcd_d12_splitchannels[0], dcd_d12_splitchannels[1]), 1)        
        
        predict_d12_scaled = F.interpolate(output_d12_2chan_interp + predict_d12, scale_factor = 2, mode = 'bilinear', align_corners = True)
        output_d12_sigmoid_interp = F.interpolate(sig_d12, scale_factor = 2, mode = 'nearest')        
        
        output_d12 = torch.cat((predict_d12_scaled, output_d12_sigmoid_interp), 1)
        
        output_d12_2chan_interp = F.interpolate(output_d12_2chan_interp, scale_factor = 2, mode = 'bilinear', align_corners = True)

        input_11 = torch.cat((input_d, predict_d12_scaled), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])
        
        dcd_d11_splitchannels = torch.split(dcd_d11[1], 1, 1)
        output_d11 = torch.cat((dcd_d11_splitchannels[0], dcd_d11_splitchannels[1]), 1)
        output_d11 = output_d11 + predict_d12_scaled
        output_d11 = torch.cat((output_d11, dcd_d11_splitchannels[2]), 1)        

        return output_d11, output_d12, output_d14


class MaxPooling(nn.Module):

    def __init__(self, kernelsize, stride):
        super(MaxPooling, self).__init__()
        if stride == 1:
            pad = int((kernelsize - 1)/2)
        else: 
            pad = 0            
        self.pool = nn.MaxPool2d(kernel_size = kernelsize, stride = stride, padding = pad)

    def forward(self, predlayer):

        predlayer = self.pool(predlayer)

        return predlayer        


class UpProject(nn.Module):

    def __init__(self, in_channels, out_channels, batch_size, statsFlag = True):
        super(UpProject, self).__init__()
        self.batch_size = batch_size

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.bn1_1 = nn.BatchNorm2d(out_channels, track_running_stats = statsFlag)
        self.bn1_2 = nn.BatchNorm2d(out_channels, track_running_stats = statsFlag)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats = statsFlag)

    def forward(self, x):
        out1_1 = self.conv1_1(nn.functional.pad(x, (1, 1, 1, 1)))

        out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github

        out1_3 = self.conv1_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github

        out1_4 = self.conv1_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        out2_1 = self.conv2_1(nn.functional.pad(x, (1, 1, 1, 1)))

        out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github

        out2_3 = self.conv2_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github

        out2_4 = self.conv2_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            self.batch_size, -1, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            self.batch_size, -1, height, width * 2)

        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            self.batch_size, -1, height * 2, width * 2)

        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            self.batch_size, -1, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            self.batch_size, -1, height, width * 2)

        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            self.batch_size, -1, height * 2, width * 2)

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.relu(out)

        return out

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, statsFlag = True):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, statsFlag = statsFlag),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, statsFlag = statsFlag)

        self.ds = convbn(inplanes, planes, 1, stride, padding = 0, statsFlag = statsFlag)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        x = self.ds(x)
        out += x
        out = self.relu(out)
        return out

