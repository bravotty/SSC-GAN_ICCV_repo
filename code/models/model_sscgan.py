import sys
import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Upsample
import pretrainedmodels
from torch.nn.utils import weight_norm


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


def convlxl(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=13, stride=1,
                     padding=1, bias=False)



# ############## M networks ################################################
class M_NET(nn.Module):
    def __init__(self):
        super(M_NET, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM  # 64
        self.in_dim = cfg.GAN.Z_DIM  # 100
        self.embed = nn.Embedding(cfg.CLASSES, self.in_dim)
        self.linear1 = nn.Linear(self.in_dim, self.in_dim, bias=False)
        self.linear2 = nn.Linear(self.in_dim, self.in_dim, bias=False)
        self.linear3 = nn.Linear(self.in_dim, self.in_dim, bias=False)
        self.linear4 = nn.Linear(self.in_dim, self.in_dim, bias=False)
        self.linear_repa = nn.Linear(self.in_dim * 2, self.in_dim * 2, bias=False)
        self.linear5 = nn.Linear(self.in_dim * 2, self.in_dim, bias=False)

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def forward(self, y_label, z_code):
        yz = self.embed(y_label)
        yz = torch.cat((z_code, yz), 1)
        z_code_new = self.linear_repa(yz)
        mu = z_code_new[:, :self.in_dim]
        logvar = z_code_new[:, self.in_dim:]
        z_repa = self.reparametrize(mu, logvar)
        return z_repa, mu, logvar


class M_NET_dec(nn.Module):
    def __init__(self):
        super(M_NET_dec, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM  # 64
        self.in_dim = cfg.GAN.Z_DIM  # 100
        self.embed = nn.Embedding(cfg.CLASSES, self.in_dim)
        self.linear1 = nn.Linear(self.in_dim * 2, self.in_dim, bias=False)
        self.linear2 = nn.Linear(self.in_dim, self.in_dim, bias=False)

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def forward(self, y_label, z_code):
        yz = self.embed(y_label)
        yz = torch.cat((z_code, yz), 1)
        z_code1 = self.linear1(yz)
        z_code2 = self.linear2(z_code1)
        return z_code2


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


def sameBlock(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, c_flag):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.c_flag = c_flag
        if self.c_flag == 1:
            self.in_dim = cfg.GAN.Z_DIM
        elif self.c_flag == 2:
            self.in_dim = cfg.GAN.Z_DIM

        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        print(ngf)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        self.upsample5 = upBlock(ngf // 16, ngf // 16)

    def forward(self, z_code):
        out_code = self.fc(z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        out_code = self.upsample5(out_code)
        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, use_hrc=1, num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if use_hrc == 1:  # For parent stage
            self.ef_dim = cfg.SUPER_CATEGORIES
        else:  # For child stage
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES

        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.samesample = sameBlock(ngf, ngf // 2)

    def forward(self, h_code):
        out_code = self.jointConv(h_code)
        out_code = self.residual(out_code)
        out_code = self.samesample(out_code)
        return out_code


class NEXT_STAGE_G_y(nn.Module):
    def __init__(self, ngf, use_hrc=1, num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G_y, self).__init__()
        self.gf_dim = ngf
        if use_hrc == 1:
            self.ef_dim = cfg.SUPER_CATEGORIES
        else:
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES

        # set num_residual from 2 to 10
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim
        num_class_side_input = cfg.CLASSES

        self.jointConv = Block3x3_relu(ngf // 2 + num_class_side_input + 50, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.samesample = sameBlock(ngf, ngf // 2)

    def forward(self, h_code, y_label, c_code):
        y_label = y_label.unsqueeze(2).unsqueeze(2)
        y_label_c = y_label.expand(y_label.size(0), y_label.size(1), h_code.size(2), h_code.size(3))
        h_c_code = torch.cat((h_code, y_label_c), 1)
        s_size = h_c_code.size(2)
        c_code = c_code.view(-1, 50, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((c_code, h_c_code), 1)
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.samesample(out_code)
        return out_code


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class GET_MASK_G(nn.Module):
    def __init__(self, ngf):
        super(GET_MASK_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 1),
            nn.Sigmoid()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.define_module()
        self.upsampling = Upsample(scale_factor=2, mode='bilinear')
        self.scale_fimg = nn.UpsamplingBilinear2d(size=[126, 126])

    def define_module(self):
        self.h_net1 = INIT_STAGE_G(self.gf_dim * 16, 1)
        self.h_net2 = NEXT_STAGE_G(self.gf_dim, use_hrc=1)
        self.h_net3 = NEXT_STAGE_G_y(self.gf_dim, use_hrc=0)
        self.img_net3 = GET_IMAGE_G(self.gf_dim // 2)

    def forward(self, y_label, z_code, c_code):
        h_code1 = self.h_net1(z_code)
        h_code2 = self.h_net2(h_code1)
        h_code3 = self.h_net3(h_code2, y_label, c_code)
        fake_img = self.img_net3(h_code3)
        return fake_img


# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


def maskBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 8, 8, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_planes, out_planes, 4, 4, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


def lastBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


def encode_y_img(ndf):
    encode_img = nn.Sequential(
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


def encode_parent_and_child_img(ndf):
    encode_img = nn.Sequential(
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


def encode_background_img(ndf):
    encode_img = nn.Sequential(
        nn.Conv2d(3, ndf, 4, 2, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
    )
    return encode_img


class D_NET(nn.Module):
    def __init__(self, stg_no):
        super(D_NET, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM  # 64
        self.stg_no = stg_no

        if self.stg_no == 0:
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES
        elif self.stg_no == 1:
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES
        elif self.stg_no == 2:
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES
        else:
            print("Invalid stage number. Set stage number as follows:")
            print("0 - for b stage")
            print("1 - for y stage")
            print("...Exiting now")
            sys.exit(0)
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim

        if self.stg_no == 0:
            self.img_code_s16 = encode_parent_and_child_img(ndf)
            self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
            self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

            self.conv = nn.Sequential(
                weight_norm(nn.Conv2d(512, 256, 3, padding=0)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv2d(256, 256, 1, padding=0)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv2d(256, 256, 1, padding=0)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.AdaptiveAvgPool2d(output_size=1)
            )
            self.fc = weight_norm(nn.Linear(256, cfg.CLASSES))

            self.jointConv = Block3x3_leakRelu(ndf * 8, ndf * 8)
            self.logits = nn.Sequential(
                nn.Conv2d(ndf * 8, efg, kernel_size=4, stride=4))

        elif self.stg_no == 1:
            self.img_code_s16 = encode_y_img(ndf)
            self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
            self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
            self.mask_block = maskBlock(3, ndf * 8)
            self.emb_visual = lastBlock(ndf * 8, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
            self.linear = nn.Linear(ndf * 8 * 16, 10)  # self.low_dim

            self.embed = nn.Embedding(cfg.CLASSES, ndf * 8 * 16)

        elif self.stg_no == 2:
            self.linear = nn.Linear(100, 64)  # self.low_dim
            self.linear1 = nn.Linear(64, 32)
            self.linear2 = nn.Linear(32, 1)

    def forward(self, x_var, y_label, mask=None):
        # for Dmagn stage
        if self.stg_no == 0:
            x_code = self.img_code_s16(x_var)
            x_code = self.img_code_s32(x_code)
            x_code = self.img_code_s32_1(x_code)
            rf_score = self.uncond_logits(x_code)

            h_c_code = self.jointConv(x_code)
            code_pred = self.logits(h_c_code)

            out = self.conv(x_code)
            out = out.view(x_code.size(0), -1)
            out = self.fc(out)
            return [code_pred.view(-1, self.ef_dim), rf_score.view(-1), out]

        # for y stage
        elif self.stg_no == 1:
            x_code = self.img_code_s16(x_var)
            x_code = self.img_code_s32(x_code)
            x_code = self.img_code_s32_1(x_code)  # [bs, 64*8, 4, 4]
            x_y_code = x_code.view(x_var.size(0), -1)  # [bs, 64 * 8 * 16]
            rf_score_real = self.uncond_logits(x_code)  # [bs, 1, 1, 1]
            rf_view = rf_score_real.view(x_var.size(0), -1)  # [bs, 1]
            out = rf_view + torch.sum(self.embed(y_label) * x_y_code, 1, keepdim=True)
            out = F.sigmoid(out)

            fea = self.linear(x_y_code)
            return out, fea

        elif self.stg_no == 2:
            x_code = self.linear(x_var)
            x_code = self.linear1(x_code)
            x_code = self.linear2(x_code)

            out = F.sigmoid(x_code)
            return out

import torchvision.models as models
import torch.utils.model_zoo as model_zoo


# ############################## For Compute inception score ##############################
# Besides the inception score computed by pretrained model, especially for fine-grained datasets (such as birds, bedroom),
#  it is also good to compute inception score using fine-tuned model and manually examine the image quality.
class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        # self.model = Inception3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # print(next(model.parameters()).data)
        state_dict = model_zoo.load_url(url, map_location=lambda storage, loc: storage)
        # pth_path = '../models_pth/inceptionv3/inception_birds.pth'
        # state_dict = torch.load(pth_path)
        self.model.load_state_dict(state_dict)

        for param in self.model.parameters():
            param.requires_grad = False
        # print('Load pretrained model from ', pth_path)
        # print(self.model)

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        x = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.model(x)
        x = nn.Softmax()(x)
        return x


def encode_imgs(ndf, in_c=3):
    encode_img = nn.Sequential(
        nn.Conv2d(in_c, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes),
                          nn.BatchNorm2d(out_planes),
                          nn.LeakyReLU(0.2, inplace=True))
    return block


class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()
        self.ndf = 64
        self.softmax = torch.nn.Softmax(dim=1)
        self.in_dim = cfg.GAN.Z_DIM
        self.model_z = nn.Sequential(encode_imgs(self.ndf),
                                     downBlock(self.ndf * 8, self.ndf * 16),
                                     Block3x3_leakRelu(self.ndf * 16, self.ndf * 8),
                                     Block3x3_leakRelu(self.ndf * 8, self.ndf * 8),
                                     nn.Conv2d(self.ndf * 8, self.in_dim, kernel_size=4, stride=4))

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def forward(self, x_var, y_label=None):
        z_x = self.model_z(x_var).view(x_var.size(0), -1)
        return z_x

################################## BI_DIS #######################################
class Gaussian(nn.Module):
    def __init__(self, std):
        super(Gaussian, self).__init__()
        self.std = std

    def forward(self, x):
        n = torch.randn_like(x) * self.std
        return x + n


class ConvT_Block(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, bn=True, activation=None, noise=False, std=None):
        super(ConvT_Block, self).__init__()
        model = [nn.ConvTranspose2d(in_c, out_c, k, s, p)]
        if bn:
            model.append(nn.BatchNorm2d(out_c))
        if activation == 'relu':
            model.append(nn.ReLU())
        elif activation == 'elu':
            model.append(nn.ELU())
        elif activation == 'leaky':
            model.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            model.append(nn.Tanh())
        elif activation == 'sigmoid':
            model.append(nn.Sigmoid())
        elif activation == 'softmax':
            model.append(nn.Softmax(dim=1))
        if noise:
            model.append(Gaussian(std))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c, k, s, p=0, bn=True, activation=None, noise=False, std=None):
        super(Conv_Block, self).__init__()
        model = [nn.Conv2d(in_c, out_c, k, s, p)]
        if bn:
            model.append(nn.BatchNorm2d(out_c))
        if activation == 'relu':
            model.append(nn.ReLU())
        if activation == 'elu':
            model.append(nn.ELU())
        if activation == 'leaky':
            model.append(nn.LeakyReLU(0.2))
        if activation == 'tanh':
            model.append(nn.Tanh())
        if activation == 'sigmoid':
            model.append(nn.Sigmoid())
        if activation == 'softmax':
            model.append(nn.Softmax(dim=1))
        if noise:
            model.append(Gaussian(std))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Linear_Block(nn.Module):
    def __init__(self, in_c, out_c, bn=True, activation=None, noise=False, std=None):
        super(Linear_Block, self).__init__()
        model = [nn.Linear(in_c, out_c)]
        if bn:
            model.append(nn.BatchNorm1d(out_c))
        if activation == 'relu':
            model.append(nn.ReLU())
        if activation == 'elu':
            model.append(nn.ELU())
        if activation == 'leaky':
            model.append(nn.LeakyReLU(0.2))
        if activation == 'tanh':
            model.append(nn.Tanh())
        if activation == 'sigmoid':
            model.append(nn.Sigmoid())
        if activation == 'softmax':
            model.append(nn.Softmax(dim=1))

        if noise:
            model.append(Gaussian(std))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Viewer(nn.Module):
    def __init__(self, shape):
        super(Viewer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Bi_Dis_base(nn.Module):
    def __init__(self, code_len, ngf=16):
        super(Bi_Dis_base, self).__init__()

        self.image_layer = nn.Sequential(
            Conv_Block(3, ngf, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.3),
            Conv_Block(ngf, ngf * 2, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5),
            Conv_Block(ngf * 2, ngf * 4, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5),
            Conv_Block(ngf * 4, ngf * 8, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5),
            Conv_Block(ngf * 8, ngf * 16, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5),
            Conv_Block(ngf * 16, 512, 4, 1, 0, bn=False, activation='leaky', noise=False, std=0.5),
            Viewer([-1, 512]))

        self.code_layer = nn.Sequential(Linear_Block(code_len, 512, bn=False, activation='leaky', noise=True, std=0.5),
                                        Linear_Block(512, 512, bn=False, activation='leaky', noise=True, std=0.3),
                                        Linear_Block(512, 512, bn=False, activation='leaky', noise=True, std=0.3))

        self.joint = nn.Sequential(Linear_Block(1024, 1024, bn=False, activation='leaky', noise=False, std=0.5),
                                   Linear_Block(1024, 1, bn=False, activation='None'),
                                   Viewer([-1]))

    def forward(self, img, code):
        t1 = self.image_layer(img)
        t2 = self.code_layer(code)
        return self.joint(torch.cat([t1, t2], dim=1))


class Bi_Dis(nn.Module):
    def __init__(self):
        super(Bi_Dis, self).__init__()
        self.BD_z = Bi_Dis_base(cfg.GAN.Z_DIM)

    def forward(self, img, z_code):
        which_pair_z = self.BD_z(img, z_code)

        return which_pair_z


# ############## C networks ################################################
class C_NET(nn.Module):
    def __init__(self):
        super(C_NET, self).__init__()
        self.in_dim = 3
        self.num_classes = cfg.CLASSES

        self.define_module()

    def define_module(self):
        # noise add layer - not implement
        self.conv1 = nn.Sequential(
            weight_norm(nn.Conv2d(3, 128, 3, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            weight_norm(nn.Conv2d(128, 256, 3, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            weight_norm(nn.Conv2d(256, 512, 3, padding=0)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            weight_norm(nn.Conv2d(512, 256, 1, padding=0)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            weight_norm(nn.Conv2d(256, 128, 1, padding=0)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AvgPool2d(6, stride=2, padding=0),
        )
        self.fc = weight_norm(nn.Linear(128 * 13 * 13, self.num_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(-1, 13 * 13 * 128)
        out = self.fc(out)
        return out


class RESNET_C(nn.Module):
    def __init__(self):
        super(RESNET_C, self).__init__()
        self.num_classes = cfg.CLASSES
        # pretrained model checkpoints
        model_name = 'resnet50'
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        # self.model = models.inception_v3(pretrained=True)
        # self.model.aux_logits = False
        # print (self.model.aux_logits, self.model.training)
        # fc_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(fc_features, self.num_classes, bias=False)

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        # x = input * 0.5 + 0.5
        # # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # # --> mean = 0, std = 1
        # x[:, 0] = (x[:, 0] - 0.485) / 0.229
        # x[:, 1] = (x[:, 1] - 0.456) / 0.224
        # x[:, 2] = (x[:, 2] - 0.406) / 0.225
        # # --> fixed-size input: batch x 3 x 299 x 299
        # x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # # 299 x 299 x 3
        #
        x = self.model(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out, x