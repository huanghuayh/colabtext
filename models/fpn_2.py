import torch
import torch.nn as nn
import torch.nn.functional as F

class conv1d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", stride=1, dilation=1):
        super().__init__()
        self.c = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.c(x)))


def upsample(input, size=None):
    upsampling = 'linear'
    upsampler = nn.Upsample(scale_factor=size, mode=upsampling)
    out = upsampler(input)
    return out
def concat(prev_output, upsampled_features):
    return torch.cat([prev_output, upsampled_features], dim=1)

class PPool(nn.Module):
    def __init__(self, in_channels, pool_size_lst=None):
        super(PPool, self).__init__()
        if pool_size_lst is None:
            pool_size_lst = [1, 2, 4]
        self.pool_size = pool_size_lst
        self.pool1 = nn.AdaptiveAvgPool1d(self.pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool1d(self.pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool1d(self.pool_size[2])
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, feat_map):
        interpolate_size = feat_map.size(-1)
        p1 = self.conv(self.pool1(feat_map))
        p2 = self.conv(self.pool2(feat_map))
        p3 = self.conv(self.pool3(feat_map))
        u1 = upsample(p1, interpolate_size)
        u2 = upsample(p2, interpolate_size // p2.size(2))
        u3 = upsample(p3, interpolate_size // p3.size(2))
        return torch.cat([feat_map, u1, u2, u3], dim=1)

class PPSP(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(PPSP, self).__init__()
        kernel_size, pad, out_channels = 3, 1, 32
        self.conv_block1 = conv1d_block(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_block3 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_block4 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_block5 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_block6 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_block7 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_block8 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_block9 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_block10 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv_block11 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_block12 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_concat = conv1d_block(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pyramid_pool = PPool(in_channels=32)
        self.dropout = nn.Dropout1d(p=0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        self.conv_upsample = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=2, stride=2)

        self.conv_output1 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_output2 = conv1d_block(in_channels=35, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv_output3 = conv1d_block(in_channels=2, out_channels=1, kernel_size=1, stride=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        ## Input block
        x = self.conv_block1(input)

        ## Downsampling block
        x1 = self.conv_block2(x)
        x1 = self.dropout(x1)
        x2 = self.max_pool(x1)

        x3 = self.conv_block3(x2)
        x3 = self.dropout(x3)
        x4 = self.max_pool(x3)

        x5 = self.conv_block4(x4)
        x5 = self.dropout(x5)
        x6 = self.max_pool(x5)

        x7 = self.conv_block5(x6)
        x7 = self.dropout(x7)
        x8 = self.max_pool(x7)

        x9 = self.conv_block6(x8)
        x9 = self.dropout(x9)
        x10 = self.max_pool(x9)

        x11 = self.conv_block7(x10)
        x11 = self.dropout(x11)
        x12 = self.max_pool(x11)

        x13 = self.conv_block8(x12)
        x13 = self.dropout(x13)
        x14 = self.max_pool(x13)

        x15 = self.conv_block9(x14)
        x15 = self.dropout(x15)
        x16 = self.max_pool(x15)

        x17 = self.conv_block10(x16)
        x17 = self.dropout(x17)
        x18 = self.max_pool(x17)

        x19 = self.conv_block10(x18)
        x19 = self.dropout(x19)
        x19 = self.conv_block10(x19)

        ## Upsampling code
        u1 = concat(x17, self.upsample(x19))
        u1 = self.conv_concat(u1)
        u1 = self.dropout(u1)

        u2 = concat(x15, self.upsample(u1))
        u2 = self.conv_concat(u2)
        u2 = self.dropout(u2)

        u3 = concat(x13, self.upsample(u2))
        u3 = self.conv_concat(u3)
        u3 = self.dropout(u3)

        u4 = concat(x11, self.upsample(u3))
        u4 = self.conv_concat(u4)
        u4 = self.dropout(u4)

        u5 = concat(x9, self.upsample(u4))
        u5 = self.conv_concat(u5)
        u5 = self.dropout(u5)

        u6 = concat(x7, self.upsample(u5))
        u6 = self.conv_concat(u6)
        u6 = self.dropout(u6)

        u7 = concat(x5, self.upsample(u6))
        u7 = self.conv_concat(u7)
        u7 = self.dropout(u7)

        u8 = concat(x3, self.upsample(u7))
        u8 = self.conv_concat(u8)
        u8 = self.dropout(u8)

        u9 = concat(x1, self.upsample(u8))
        u9 = self.conv_concat(u9)
        u9 = self.dropout(u9)

        ## Output block

        output1 = self.conv_output1(u9)
        p_pool_out = self.pyramid_pool(output1)
        output2 = self.conv_output2(p_pool_out)

        output3 = self.conv_output3(output2)
        # final_output = self.sigmoid(output3)

        return output3


# x=torch.randn(1, 1, 1024)
# model = PPSP(in_channels=1, out_channels=32)
# out1,out2 = model(x)
# print()