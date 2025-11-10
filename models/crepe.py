import torch.nn as nn
import torch
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


class DenseHead(nn.Module):
    def __init__(self, in_flat, hidden, out_len):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_flat, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_len)
        )
    def forward(self, f_flat):
        return self.fc(f_flat).unsqueeze(1)

class FPN_2_mtl(nn.Module):
    def __init__(self, in_channels=1, out_channels=32,padding="same", stride=1, dilation=1):
        super(FPN_2_mtl, self).__init__()
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
        # self.pyramid_pool = PPool(in_channels=32)
        self.dropout = nn.Dropout1d(p=0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        self.conv_upsample = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=2, stride=2)

        self.conv_output1 = conv1d_block(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_output2 = conv1d_block(in_channels=35, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv_output3 = conv1d_block(in_channels=2, out_channels=1, kernel_size=1, stride=1)

        self.sigmoid = nn.Sigmoid()

        # --- Dense layer after encoder ---
        bottleneck_len = 1024 // (2 ** 5)
        dense_in_features = 32 * bottleneck_len
        self.flatten = nn.Flatten()
        self.head1 = DenseHead(dense_in_features, hidden=dense_in_features*4, out_len=1024)

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

        # x11 = self.conv_block7(x10)
        # x11 = self.dropout(x11)
        # x12 = self.max_pool(x11)
        #
        # x13 = self.conv_block8(x12)
        # x13 = self.dropout(x13)
        # x14 = self.max_pool(x13)
        #
        # x15 = self.conv_block9(x14)
        # x15 = self.dropout(x15)
        # x16 = self.max_pool(x15)
        #
        # x17 = self.conv_block10(x16)
        # x17 = self.dropout(x17)
        # x18 = self.max_pool(x17)
        #
        x19 = self.conv_block10(x10)
        x19 = self.dropout(x19)
        x19 = self.conv_block10(x19)

        flatten = self.flatten(x19)
        final_output = self.head1(flatten)

        return final_output


