import torch, os
import torch.nn as nn
from torch.nn import functional as F
import time

class conv_block_3d(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(conv_block_3d, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d( out_dim)
        self.act = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def channel_shuffle(x, group):
    batchsize, num_channels, depth, height, width = x.data.size()
    assert num_channels % group == 0
    group_channels = num_channels // group

    x = x.reshape(batchsize, group_channels, group, depth, height, width)
    x = x.permute(0, 2, 1, 3, 4, 5)
    x = x.reshape(batchsize, num_channels, depth, height, width)
    return x



class sep_conv_block_3d(nn.Module):
    def __init__(self, in_dim, out_dim, shuffle, add_channel):
        super(sep_conv_block_3d, self).__init__()
        self.shuffle = shuffle
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        self.add_channel = add_channel
        self.conv_133 = nn.Conv3d(self.inter_dim, self.out_inter_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
        self.bn_layer_1 = nn.InstanceNorm3d(self.out_inter_dim)

        self.conv_313 = nn.Conv3d(self.inter_dim, self.out_inter_dim, kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1), bias=False)
        self.bn_layer_2 = nn.InstanceNorm3d(self.out_inter_dim)

        self.conv_331 = nn.Conv3d(self.inter_dim, self.out_inter_dim, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=False)
        self.bn_layer_3 = nn.InstanceNorm3d(self.out_inter_dim)

        self.pool = max_pooling_3d_noz()
        self.conv_333 = nn.Conv3d(self.inter_dim, self.out_inter_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
        self.bn_layer_4 = nn.InstanceNorm3d(self.out_inter_dim)

        self.conv_1x1 = nn.Conv3d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)

        if add_channel == 1:
            self.conv12_1x1 = nn.Conv3d(self.out_inter_dim, self.inter_dim, kernel_size=1, stride=1, padding=0)
            self.conv23_1x1 = nn.Conv3d(self.out_inter_dim, self.inter_dim, kernel_size=1, stride=1, padding=0)
            self.conv34_1x1 = nn.Conv3d(self.out_inter_dim, self.inter_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.shuffle == 1:
            x = channel_shuffle(x, 4)
        x1 = x[:, 0:self.inter_dim, ...]
        x2 = x[:, self.inter_dim:self.inter_dim * 2, ...]
        x3 = x[:, self.inter_dim * 2:self.inter_dim * 3, ...]
        x4 = x[:, self.inter_dim * 3:self.inter_dim * 4, ...]

        x1 = self.conv_133(x1)
        x1 = self.bn_layer_1(x1)

        if self.add_channel == 1:
            x2 = self.conv12_1x1(x1) + x2
        else:
            x2 = x1 + x2
        x2 = self.conv_313(x2)
        x2 = self.bn_layer_2(x2)

        if self.add_channel == 1:
            x3 = self.conv23_1x1(x2) + x3
        else:
            x3 = x2 + x3
        x3 = self.conv_331(x3)
        x3 = self.bn_layer_3(x3)

        if self.add_channel == 1:
            x4 = self.conv34_1x1(x3) + x4
        else:
            x4 = x3 + x4
        x4 = self.pool(x4)
        x4 = self.conv_333(x4)
        # up_op = nn.Upsample(size=(x4.shape[2], x3.shape[3], x3.shape[4]))
        # x4 = up_op(x4)  # -> [1, 16, 64, 64, 64]
        x4 = F.interpolate(x4, size=(x4.shape[2], x3.shape[3], x3.shape[4]))
        x4 = self.bn_layer_4(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)  # -> [1, 48, 32, 32, 32]
        x = self.conv_1x1(x)
        return x


class double_block(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(double_block, self).__init__()
        self.conv1 = sep_conv_block_3d(in_dim, out_dim, 0, 1)
        self.bn_layer_1 = nn.InstanceNorm3d(out_dim)
        self.activation_1 = activation

        self.conv2 = sep_conv_block_3d(out_dim, out_dim, 1, 0)
        self.bn_layer_2 = nn.InstanceNorm3d(out_dim)
        self.activation_2 = activation

        self.conv_1x1_resdial = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn_layer_1(x1)
        x1 = self.activation_1(x1)

        x1 = self.conv2(x1) + self.conv_1x1_resdial(x)
        x1 = self.bn_layer_1(x1)
        x1 = self.activation_1(x1)
        return x1


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        self.prelu = nn.PReLU()

        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, g, x):
        x_sub = F.interpolate(x, size=(g.shape[2], g.shape[3], g.shape[4]), mode='trilinear', align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x_sub)
        psi = self.prelu(g1 + x1)
        psi = self.psi(psi)
        x_sub = x_sub * psi

        x_sub = F.interpolate(x_sub, size=(x.shape[2], x.shape[3], x.shape[4]), mode='trilinear', align_corners=True)
        out = x_sub + self.beta * x
        return out


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        nn.InstanceNorm3d( out_dim),
        activation, )


def max_pooling_3d_noz():
    # this pooling don't change the number of z axis
    return nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim // 2, activation),
        conv_block_3d(out_dim // 2, out_dim, activation)
    )


class UNet_3D(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet_3D, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_f = num_filters
        activation = nn.PReLU()
        #nn.PReLU
        # Down sampling path
        self.encoder_level_1_1 = conv_block_3d(self.in_dim, self.n_f, activation)
        self.encoder_level_1_2 = sep_conv_block_3d(self.n_f, self.n_f, activation, 0)
        self.norm_layer_1 = nn.InstanceNorm3d(self.n_f)
        self.activation_1 = nn.PReLU()
        # self.encoder_level_2 = double_block(self.in_dim, self.n_f, activation)
        self.pool_1 = max_pooling_3d()

        self.encoder_level_2 = double_block(self.n_f, self.n_f * 2, activation)
        self.pool_2 = max_pooling_3d()

        self.encoder_level_3 = double_block(self.n_f * 2, self.n_f * 4, activation)
        self.pool_3 = max_pooling_3d_noz()

        # Bridge
        self.bridge = double_block(self.n_f * 4, self.n_f * 8, activation)
        self.level3_1x1 = nn.Conv3d(self.n_f * 8, self.n_f * 4, kernel_size=1, stride=1, padding=0)
        self.decoder_level_3 = double_block(self.n_f * 4, self.n_f * 4, activation)

        self.level2_1x1 = nn.Conv3d(self.n_f * 4, self.n_f * 2, kernel_size=1, stride=1, padding=0)
        self.decoder_level_2 = double_block(self.n_f * 2, self.n_f * 2, activation)

        self.level1_1x1 = nn.Conv3d(self.n_f * 2, self.n_f * 1, kernel_size=1, stride=1, padding=0)
        self.decoder_level_1 = double_block(self.n_f * 1, self.n_f * 1, activation)

        # Up sampling path
        self.level3_1x1_reg = nn.Conv3d(self.n_f * 4, self.n_f * 2, kernel_size=1, stride=1, padding=0)
        self.level3_1x1_reg_l3 = nn.Conv3d(self.n_f * 4, self.n_f * 2, kernel_size=1, stride=1, padding=0)
        self.decoder_level_3_reg = double_block(self.n_f * 2, self.n_f * 2, activation)

        self.level2_1x1_reg = nn.Conv3d(self.n_f * 2, self.n_f * 1, kernel_size=1, stride=1, padding=0)
        self.level2_1x1_reg_l2 = nn.Conv3d(self.n_f * 2, self.n_f * 1, kernel_size=1, stride=1, padding=0)
        self.decoder_level_2_reg = double_block(self.n_f * 1, self.n_f * 1, activation)

        self.level1_1x1_reg = nn.Conv3d(self.n_f, self.n_f // 2, kernel_size=1, stride=1, padding=0)
        self.level1_1x1_reg_l1 = nn.Conv3d(self.n_f, self.n_f // 2, kernel_size=1, stride=1, padding=0)
        self.decoder_level_1_reg = double_block(self.n_f // 2, self.n_f // 2, activation)

        # Output
        self.out_1 = double_block(96, self.n_f//2, activation)

        self.out = nn.Conv3d(self.n_f//2, out_dim, kernel_size=1, stride=1, padding=0)

        self.out_reg = nn.Conv3d(self.n_f//2, 1, kernel_size=1, stride=1, padding=0)
        self.out_reg_sigmod = nn.Sigmoid()

    def forward(self, x):
        # Down sampling
        x1 = self.encoder_level_1_1(x)
        x1 = self.encoder_level_1_2(x1)
        x1 = self.norm_layer_1(x1)
        x1 = self.activation_1(x1)
        x = self.pool_1(x1)
        x2 = self.encoder_level_2(x)
        x = self.pool_2(x2)

        x3 = self.encoder_level_3(x)
        x = self.pool_3(x3)

        # Bridge
        x = self.bridge(x)

        # Up sampling for segmentation
        x = F.interpolate(x, size=(x3.shape[2], x3.shape[3], x3.shape[4]), mode='trilinear', align_corners=True)
        x = self.level3_1x1(x)

        xr = x
        xr = self.level3_1x1_reg(xr)

        x = x + x3
        x = self.decoder_level_3(x)

        x = F.interpolate(x, size=(x2.shape[2], x2.shape[3], x2.shape[4]), mode='trilinear', align_corners=True)

        x = self.level2_1x1(x)
        x = x + x2
        x = self.decoder_level_2(x)

        x = F.interpolate(x, size=(x1.shape[2], x1.shape[3], x1.shape[4]), mode='trilinear', align_corners=True)
        x = self.level1_1x1(x)
        x = x + x1
        x = self.decoder_level_1(x)

        # Up sampling for regression
        x3 = self.level3_1x1_reg_l3(x3)
        xr = xr + x3
        xr = self.decoder_level_3_reg(xr)

        xr = F.interpolate(xr, size=(x2.shape[2], x2.shape[3], x2.shape[4]), mode='trilinear', align_corners=True)
        xr = self.level2_1x1_reg(xr)
        x2 = self.level2_1x1_reg_l2(x2)
        xr = xr + x2
        xr = self.decoder_level_2_reg(xr)

        xr = F.interpolate(xr, size=(x1.shape[2], x1.shape[3], x1.shape[4]), mode='trilinear', align_corners=True)
        xr = self.level1_1x1_reg(xr)
        x1 = self.level1_1x1_reg_l1(x1)
        xr = xr + x1
        xr = self.decoder_level_1_reg(xr)

        # Output
        x = torch.cat([x, xr], dim=1)

        xr = self.out_reg(xr)
        xr = self.out_reg_sigmod(xr)

        x = self.out_1(x)
        x = self.out(x)

        return x, xr


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_size = 88
    x = torch.Tensor(1, 3, image_size, image_size, image_size)
    x = torch.Tensor(1, 1, 20, 180, 180)

    # x.to(device)
    print("x size: {}".format(x.size()))

    model = UNet_3D(in_dim=1, out_dim=2, num_filters=64)
    print(model)
    num_para = 0
    for para in model.parameters():
        num_para += para.numel()
    print(num_para)
    t1 = time.clock()
    out, out_reg = model(x)
    t2 = time.clock()
    print('cost time:', t2-t1)
    print("out size: {}".format(out.size()))
    print("out reg size: {}".format(out_reg.size()))