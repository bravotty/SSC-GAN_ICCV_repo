from __future__ import print_function
from six.moves import range
import sys
import shutil
import numpy as np
import os

import random
import time
from PIL import Image
from copy import deepcopy

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
from itertools import repeat, cycle
from torch.nn.functional import softmax, log_softmax
from torch.nn.functional import cosine_similarity
from tensorboardX import summary
from tensorboardX import FileWriter
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from miscc.config import cfg
from miscc.utils import mkdir_p
from torch.optim import lr_scheduler
from models.model_sscgan import *
from trainers import tri_loss

dir = './log'
writer = SummaryWriter(dir)

# ################## Shared functions ###################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_network():
    netM = M_NET()
    netM.apply(weights_init)

    netM_dec = M_NET_dec()
    netM_dec.apply(weights_init)

    netG = G_NET()
    netG.apply(weights_init)

    netBD = Bi_Dis()
    # netBD.apply(weights_init)

    netE = Encoder()
    netE.apply(weights_init)

    netC = RESNET_C()

    netsD = []
    for i in range(3):  # 3 discriminators for background, parent and child stage
        netsD.append(D_NET(i))

    for i in range(len(netsD)):
        netsD[i].apply(weights_init)

    count = 0

    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1

    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s_%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)

    if cfg.CUDA:
        netG.cuda()
        netC.cuda()
        netM.cuda()
        netM_dec.cuda()
        netE.cuda()
        netBD.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()

    return netM, netM_dec, netE, netBD, netG, netC, netsD, len(netsD), count


def define_optimizers(netBD, netE, netM, netM_dec, netG, netC, netsD):
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    optimizerBD = optim.Adam(netBD.parameters(), lr=2e-4, betas=(0.5, 0.999))

    optimizerM = []
    optimizerM.append(optim.Adam(netM.parameters(),
                                 lr=cfg.TRAIN.GENERATOR_LR,
                                 betas=(0.5, 0.999)))
    optimizerM.append(optim.Adam(netM_dec.parameters(),
                                 lr=cfg.TRAIN.GENERATOR_LR,
                                 betas=(0.5, 0.999)))
    optimizerG = []
    optimizerG.append(optim.Adam(netG.parameters(),
                                 lr=cfg.TRAIN.GENERATOR_LR,
                                 betas=(0.5, 0.999)))
    optimizerG.append(optim.Adam(netE.parameters(),
                                 lr=cfg.TRAIN.GENERATOR_LR,
                                 betas=(0.5, 0.999)))
    optimizerG.append(optim.Adam([{'params': netsD[0].jointConv.parameters()}, {'params': netsD[0].logits.parameters()}],
                                 lr=cfg.TRAIN.GENERATOR_LR,
                                 betas=(0.5, 0.999)))
    optimizerG_mask = optim.Adam(netG.h_net3.parameters(),
                                 lr=cfg.TRAIN.GENERATOR_LR,
                                 betas=(0.5, 0.999))

    optimizerC = []
    ignored_params = list(map(id, netC.classifier.parameters()))
    print('the num of new layers:', len(ignored_params), flush=True)
    base_params = filter(lambda p: id(p) not in ignored_params, netC.parameters())
    opt = optim.SGD(
        [{'params': base_params}, {'params': netC.classifier.parameters(), 'lr': cfg.TRAIN.CLASSIFIER_LR}], \
        lr=cfg.TRAIN.CLASSIFIER_LR,
        momentum=0.9)
    optimizerC.append(opt)

    return optimizerBD, optimizerM, optimizerG, optimizerC, optimizerG_mask, optimizersD

def save_model(netM, netG, netE, avg_param_G, netC, netsD, epoch, model_dir):
    load_params(netG, avg_param_G)
    torch.save(
        netM.state_dict(),
        '%s/netM_%d.pth' % (model_dir, epoch))
    torch.save(
        netE.state_dict(),
        '%s/netE_%d.pth' % (model_dir, epoch))
    torch.save(
        netG.state_dict(),
        '%s/netG_%d.pth' % (model_dir, epoch))
    torch.save(
        netC.state_dict(),
        '%s/netC_%d.pth' % (model_dir, epoch))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(
            netD.state_dict(),
            '%s/netD%d.pth' % (model_dir, i))
    print('Save G/Ds models.')

def save_img_results(imgs_tcpu, fake_imgs, num_imgs, count, image_dir, summary_writer):
    num = cfg.TRAIN.VIS_COUNT

    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/real_samples%09d.png' % (image_dir, count),
        normalize=True)
    real_img_set = vutils.make_grid(real_img).numpy()
    real_img_set = np.transpose(real_img_set, (1, 2, 0))
    real_img_set = real_img_set * 255
    real_img_set = real_img_set.astype(np.uint8)

    for i in range(len(fake_imgs)):
        fake_img = fake_imgs[i]
        vutils.save_image(
            fake_img.data, '%s/count_%09d_fake_samples%d.png' %
                           (image_dir, count, i), normalize=True)
        fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()
        fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
        fake_img_set = (fake_img_set + 1) * 255 / 2
        fake_img_set = fake_img_set.astype(np.uint8)
        summary_writer.flush()

class SSCGAN_train(object):
    def __init__(self, output_dir, label, unlabel, test, imsize):
        # report.export_sources(os.path.join(output_dir, 'Src'))
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            self.tsne_dir = os.path.join(output_dir, 'Tsne')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            mkdir_p(self.tsne_dir)
            self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.num_classes = cfg.CLASSES
        # self.alpha_cm = cfg.TRAIN.ALPHA

        self.label_data = label
        self.unlabel_data = unlabel
        self.test_data = test
        self.num_batches = len(self.unlabel_data)

    def prepare_data(self, data):
        fimgs, cimgs, c_code, _, warped_bbox, digit_label = data
        real_vfimgs, real_vcimgs = [], []
        if cfg.CUDA:
            vc_code = Variable(c_code).cuda()
            for i in range(len(warped_bbox)):
                warped_bbox[i] = Variable(warped_bbox[i]).float().cuda()
        else:
            vc_code = Variable(c_code)
            for i in range(len(warped_bbox)):
                warped_bbox[i] = Variable(warped_bbox[i])

        if cfg.CUDA:
            real_vfimgs.append(Variable(fimgs[0]).cuda())
            real_vcimgs.append(Variable(cimgs[0]).cuda())
        else:
            real_vfimgs.append(Variable(fimgs[0]))
            real_vcimgs.append(Variable(cimgs[0]))

        return fimgs, real_vfimgs, real_vcimgs, vc_code, warped_bbox, digit_label

    def train_Dnet(self, idx):
        if idx == 0 or idx == 1 or idx == 2:
            criterion, criterion_one, criterion_class = self.criterion, self.criterion_one, self.criterion_class
            netD, optD = self.netsD[idx], self.optimizersD[idx]
            real_imgs = self.real_cimgs[0]
            real_imgs_unlabel = self.real_cimgs_unlabel[0]
            fake_imgs = self.fake_img
            # random y + fake z
            fake_imgs_fake_z = self.fake_img_fake_z
            # forward
            if idx == 0:
                netD.zero_grad()
                real_logits = netD(real_imgs, self.label_digit)
                real_logits_unlabel = netD(real_imgs_unlabel, self.label_digit)
                fake_logits = netD(fake_imgs.detach(), self.label_digit)
                fake_logits_enc_real = netD(fake_imgs_fake_z.detach(), self.u2_label_digit)
                real_labels = torch.ones_like(real_logits[1])
                fake_labels = torch.zeros_like(real_logits[1])

            if idx == 1:
                netD.zero_grad()
                real_logits, fea_label = netD(real_imgs, self.label_digit)
                fake_logits, fea_fake = netD(fake_imgs.detach(), self.label_digit)
                fake_logits_enc_real, fea_fake = netD(fake_imgs_fake_z.detach(), self.u2_label_digit)
                real_labels = torch.ones_like(real_logits)
                fake_labels = torch.zeros_like(real_logits)

            if idx == 2:
                # forward + loss
                netD.zero_grad()
                real_logits = netD(self.noise, self.label_digit)
                fake_logits = netD(self.fake_z, self.label_digit)
                real_labels = torch.ones_like(real_logits)
                fake_labels = torch.zeros_like(real_logits)

                errD_real = criterion_one(real_logits, real_labels)
                errD_fake = criterion_one(fake_logits, fake_labels)
                errD = errD_real + errD_fake

            # loss
            if idx == 0:
                errD_real = criterion_one(real_logits[1], real_labels)
                errD_fake = criterion_one(fake_logits[1], fake_labels)
                errD_fake_enc_real = criterion_one(fake_logits_enc_real[1], fake_labels)
                errD_real_unlabel = criterion_one(real_logits_unlabel[1], real_labels)

                Y_c, _ = self.netC(real_imgs_unlabel)
                Ypersudo = torch.argmax(F.softmax(Y_c.detach(), dim=1), dim=1).detach()
                persudo_y_logits = netD(real_imgs_unlabel, Ypersudo)

                errD_class = criterion_class(real_logits[2], self.label_digit) + criterion_class(fake_logits[2], self.label_digit) + \
                             criterion_class(persudo_y_logits[2], Ypersudo)

                errD = errD_real + errD_fake + errD_real_unlabel + errD_fake_enc_real + errD_class

            if idx == 1:
                errD_real = criterion_one(real_logits, real_labels)  # Real/Fake loss for the real image
                errD_fake = criterion_one(fake_logits, fake_labels)  # Real/Fake loss for the fake image
                errD_fake_enc_real = criterion_one(fake_logits_enc_real, fake_labels)

                Y_c, _ = self.netC(real_imgs_unlabel)
                Ypersudo = torch.argmax(F.softmax(Y_c.detach(), dim=1), dim=1).detach()
                persudo_y_logits, fea_unlabel = netD(real_imgs_unlabel, Ypersudo)
                real_labels = torch.ones_like(persudo_y_logits)
                errD_persudo = criterion_one(persudo_y_logits, real_labels)

                ''' triplet part '''
                fea_real = torch.cat([fea_label, fea_unlabel], dim=0)
                y_concat_real = torch.cat([self.label_digit, Ypersudo], dim=0)
                errD_triplet = 1.0 * tri_loss.triplet_loss(fea_real, y_concat_real, 0.5, 'r', 0)
                # errD_triplet = 1.0*tri_loss.triplet_loss_fake(fea_real, fea_fake, y_concat_real, 0.5, 'r', 0)
                errD = errD_real + errD_fake + errD_persudo + errD_fake_enc_real + errD_triplet
            errD.backward()
            optD.step()
        return errD

    def train_Gnet_cycle(self):
        self.netE.zero_grad()
        self.netG.zero_grad()
        # Encoder
        self.zx = self.netE(self.input_var, self.label_digit)
        self.xzx = self.netG(self.one_hot_label_random, self.zx, self.c_code)
        self.zxzx = self.netE(self.xzx, self.label_digit)

        # Cycle loss
        errG_cycle_real = self.criterion_l1loss(self.zx, self.zxzx)

        errG_cycle_real.backward()
        self.optimizerG[0].step()
        self.optimizerG[1].step()
        return errG_cycle_real


    def train_Gnet(self):
        self.netE.zero_grad()
        self.netM.zero_grad()
        self.netM_dec.zero_grad()
        self.netG.zero_grad()
        self.netC.zero_grad()
        for myit in range(len(self.netsD)):
            self.netsD[myit].zero_grad()
        criterion_one    = self.criterion_one
        criterion_class  = self.criterion_class

        # Encoder
        self.fake_z = self.netE(self.input_var, self.label_digit)
        self.fake_img_fake_z = self.netG(self.one_hot_label_random, self.fake_z, self.c_code)

        # MineGAN
        self.fake_img = self.netG(self.one_hot_label, self.noise, self.c_code)

        # fool BD loss
        pred_enc_z = self.netBD(self.input_var, self.fake_z)
        pred_gen_z = self.netBD(self.fake_img, self.noise)
        fool_BD_loss = (pred_enc_z.mean()) - (pred_gen_z.mean())

        # semantic and feature matching loss
        fake_pred, feat_xz = self.netC(self.fake_img)
        fake_pred_2, feat_xz_2 = self.netC(self.fake_img_fake_z)

        errG_ce = criterion_class(fake_pred, self.label_digit)
        errG_ce_2 = criterion_class(fake_pred_2, self.u2_label_digit)
        errG_semantic = errG_ce + errG_ce_2
        # D_overall loss
        outputs = self.netsD[0](self.fake_img, self.label_digit)
        real_labels = torch.ones_like(outputs[1])
        errG_Dmagn_fake = criterion_one(outputs[1], real_labels)

        outputs = self.netsD[0](self.fake_img_fake_z, self.u2_label_digit)
        errG_Dmagn_fake_z_rep = criterion_one(outputs[1], real_labels)
        errG_D_magn = errG_Dmagn_fake + errG_Dmagn_fake_z_rep
        # Dz
        output  = self.netsD[2](self.fake_z, self.label_digit)
        errG_Dz_fake = criterion_one(output, real_labels)
        errG_Dz = errG_Dz_fake
        # D_y loss
        outputs, _ = self.netsD[1](self.fake_img, self.label_digit)
        real_labels = torch.ones_like(outputs)
        errG_Dy_fake = criterion_one(outputs, real_labels)
        outputs, _ = self.netsD[1](self.fake_img_fake_z, self.u2_label_digit)
        errG_Dy_fake_z_rep = criterion_one(outputs, real_labels)
        errG_Dy = errG_Dy_fake + errG_Dy_fake_z_rep

        # D_overall info loss
        pred_c = self.netsD[0](self.fake_img, self.label_digit)
        errG_info_dis = criterion_class(pred_c[0], torch.nonzero(self.c_code.long())[:, 1])
        # Cycle loss

        errG_total = errG_semantic * 3 + errG_D_magn + errG_Dy + errG_Dz + errG_info_dis + fool_BD_loss
        errG_total.backward()
        self.optimizerG[0].step()
        self.optimizerG[1].step()
        self.optimizerG[2].step()
        return errG_total, errG_ce, errG_ce_2, errG_Dmagn_fake, errG_Dmagn_fake_z_rep, \
                       errG_Dy_fake, errG_Dy_fake_z_rep, errG_Dz

    def train_Cnet(self):
        self.netC.zero_grad()
        criterion_class, criterion_one, criterion_mse = self.criterion_class, self.criterion_one, self.criterion_mse

        unlabel_prediction, _ = self.netC(self.real_cimgs_unlabel[0])
        unlabel_prediction_digit = torch.argmax(F.softmax(unlabel_prediction, dim=1), dim=1)
        x_mix_unlabel, y_mix, self.lam = self.cutmix_data_between(self.real_cimgs[0], self.label_digit,
                                                                  self.real_cimgs_unlabel[0],
                                                                  unlabel_prediction_digit,
                                                                  alpha=0.2)

        unlabel_mix_pred, _ = self.netC(x_mix_unlabel)
        loss_unlabel = criterion_class(unlabel_mix_pred, self.label_digit) * self.lam + \
                       criterion_class(unlabel_mix_pred, unlabel_prediction_digit) * (1. - self.lam)

        # real loss
        pred_real, _ = self.netC(self.real_cimgs[0])
        loss_real = criterion_class(pred_real, self.label_digit.cuda())

        # temporal-ensemble loss
        self.outputs[self.j * self.batch_size: (self.j + 1) * self.batch_size] = pred_real.data.clone()
        te_loss = criterion_mse(self.zcomp, unlabel_prediction)

        errC = loss_real + loss_unlabel + self.w * te_loss
        errC.backward()
        self.optimizerC[0].step()
        return errC

    def update_ema_variables(self, netC, netC_ema, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(netC_ema.parameters(), netC.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def calc_metrics_C(self, modelC, modelD, loader):
        total_C = 0
        correct_C = 0
        correct_fake = 0
        noise = Variable(torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM)).cuda()

        for i, data in enumerate(loader):
            noise.data.normal_(0, 1)
            u1, u2, real_cimgs, c_code, u4, label_digit = self.prepare_data(data)
            label_digit = label_digit.cuda()
            label_one_hot = self.get_float_one_hot(label_digit)
            # fakeimg = self.netG(label_one_hot, noise, c_code)
            _, _, output_fake_d = modelD(real_cimgs[0], label_digit)

            # output_fake, _ = modelC(fakeimg)

            _, predicted_fake = torch.max(output_fake_d.data, 1)
            correct_fake += (predicted_fake == label_digit.data.view_as(predicted_fake)).sum()

            output_C, _ = modelC(real_cimgs[0])
            _, predicted_C = torch.max(output_C.data, 1)
            total_C += label_digit.size(0)
            correct_C += (predicted_C == label_digit.data.view_as(predicted_C)).sum()

        acc = 100 * float(correct_C) / total_C
        acc_fake = 100 * float(correct_fake) / total_C

        return acc, acc_fake


    def get_float_one_hot(self, label):
        digit_2_onehot = torch.zeros([self.batch_size, cfg.CLASSES])
        for i in range(self.batch_size):
            digit_2_onehot[i][label[i]] = 1
        digit_2_onehot = digit_2_onehot.float()
        digit_2_onehot = digit_2_onehot.cuda()
        return digit_2_onehot

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        # uniform
        # cx = np.random.randint(W)
        # cy = np.random.randint(H)
        try:
            cx = np.random.randint(low=cut_w // 2,
                                   high=W - (cut_w // 2) + 1)  # or low=(cut_w//2) - 1, high=W - (cut_w//2)
            cy = np.random.randint(low=cut_h // 2, high=H - (cut_h // 2) + 1)
        except:
            print('lam:', lam)
            print('W:', W, 'cut_w:', cut_w)
            print('H:', H, 'cut_h:', cut_h)
            print('low:', cut_w // 2, 'high:', W - cut_w // 2)
            print('low:', cut_h // 2, 'high:', H - cut_h // 2)
            exit(0)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix_data_between(self, x1, y1, x2, y2, alpha=1.0):
        '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x1.size(), lam)
        x = x1.clone()
        x[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2].data
        y = lam * y1 + (1 - lam) * y2
        mixed_x = Variable(x.cuda())
        mixed_y = Variable(y.cuda())
        return mixed_x, mixed_y, lam

    def cutmix_criterion(self, y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def ramp_up(self, epoch, max_epochs, max_val, mult):
        if epoch == 0:
            return 0.
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)

    def weight_schedule(self, epoch, max_epochs, max_val, mult, n_labeled, n_samples):
        max_val = max_val * (float(n_labeled) / n_samples)
        return self.ramp_up(epoch, max_epochs, max_val, mult)

    def cal_gradient_penalty(self, netD, real_data, fake_data, type='mixed', constant=1.0):
        # adapted from cyclegan
        """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

        Arguments:
            netD (network)              -- discriminator network
            real_data (tensor array)    -- real images
            fake_data (tensor array)    -- generated images from the generator
            device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
            type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
            constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2


        Returns the gradient penalty loss
        """

        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            interpolatesv = []
            for i in range(len(real_data)):
                alpha = torch.rand(real_data[i].shape[0], 1)
                alpha = alpha.expand(real_data[i].shape[0],
                                     real_data[i].nelement() // real_data[i].shape[0]).contiguous().view(
                    *real_data[i].shape)
                alpha = alpha.cuda()
                interpolatesv.append(alpha * real_data[i] + ((1 - alpha) * fake_data[i]))
        else:
            raise NotImplementedError('{} not implemented'.format(type))

        # require grad
        for i in range(len(interpolatesv)):
            interpolatesv[i].requires_grad_(True)

        # feed into D
        disc_interpolates = netD(*interpolatesv)

        # cal penalty
        gradient_penalty = 0
        for i in range(len(disc_interpolates)):
            for j in range(len(interpolatesv)):
                gradients = torch.autograd.grad(outputs=disc_interpolates[i], inputs=interpolatesv[j],
                                                grad_outputs=torch.ones(disc_interpolates[i].size()).cuda(),
                                                create_graph=True, retain_graph=True, only_inputs=True,
                                                allow_unused=True)
                if gradients[0] is not None:  # it will return None if input is not used in this output (allow unused)
                    gradients = gradients[0].view(real_data[j].size(0), -1)  # flat the data
                    gradient_penalty += (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean()  # added eps

        return gradient_penalty

    def train_BD(self):
        self.optimizerBD.zero_grad()
        # make prediction on pairs
        # print (self.fake_img_fake_z_rep.shape, self.fake_z.shape)
        pred_enc_z = self.netBD(self.input_var, self.fake_z.detach())
        pred_gen_z = self.netBD(self.fake_img.detach(), self.noise)

        real_data = [self.input_var, self.fake_z.detach()]
        fake_data = [self.fake_img.detach(), self.noise]

        penalty = self.cal_gradient_penalty(self.netBD, real_data, fake_data, type='mixed', constant=1.0)

        D_loss = -(pred_enc_z.mean()) + (pred_gen_z.mean()) + penalty * 10
        D_loss.backward()
        self.optimizerBD.step()

    def train(self):
        self.mtype = 'z_repa'
        self.netM, self.netM_dec, self.netE, self.netBD, self.netG, self.netC, self.netsD, self.num_Ds, start_count = load_network()
        avg_param_G = copy_G_params(self.netG)

        self.optimizerBD, self.optimizerM, self.optimizerG, self.optimizerC, self.opt_mask, self.optimizersD = \
            define_optimizers(self.netBD, self.netE, self.netM, self.netM_dec, self.netG, self.netC, self.netsD)

        self.criterion = nn.BCELoss(reduce=False)
        self.criterion_one = nn.BCELoss()
        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1loss = nn.L1Loss()

        self.real_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(0))
        nz = cfg.GAN.Z_DIM
        self.noise = Variable(torch.FloatTensor(self.batch_size, nz))
        self.noise_new = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = \
            Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))

        if cfg.CUDA:
            self.criterion.cuda()
            self.criterion_one.cuda()
            self.criterion_class.cuda()
            self.criterion_mse.cuda()
            self.criterion_l1loss.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            self.noise, fixed_noise, self.noise_new = self.noise.cuda(), fixed_noise.cuda(), self.noise_new.cuda()

        print("Starting normal SSC-GAN training..")
        count = start_count
        start_epoch = start_count // (self.num_batches)
        self.global_step = 0
        exp_lr_scheduler = lr_scheduler.StepLR(self.optimizerC[0], step_size=20, gamma=0.5)
        n_classes = cfg.CLASSES
        n_samples = len(self.unlabel_data) * self.batch_size
        Z = torch.zeros(n_samples, n_classes).float().cuda()
        z = torch.zeros(n_samples, n_classes).float().cuda()
        self.outputs = torch.zeros(n_samples, n_classes).float().cuda()
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            exp_lr_scheduler.step(epoch)
            w = self.weight_schedule(epoch, self.max_epoch, max_val=30., mult=-5., n_labeled=3000, n_samples=n_samples)
            self.w = torch.autograd.Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)
            self.j = 0
            for (data_label), (data_unlabel) in zip(cycle(self.label_data), self.unlabel_data):
                compare = []
                self.imgs_tcpu, self.real_fimgs, self.real_cimgs, \
                self.c_code, self.warped_bbox, self.label_digit = self.prepare_data(data_label)

                self.imgs_tcpu_unlabel, self.real_fimgs_unlabel, self.real_cimgs_unlabel, \
                self.c_code_unlabel, self.warped_bbox_unlabel, u2 = self.prepare_data(data_unlabel)

                self.input_var = torch.autograd.Variable(self.real_cimgs[0].cuda())
                self.u2_label_digit = u2.cuda()
                self.one_hot_label = self.get_float_one_hot(self.label_digit)
                self.one_hot_label_random = self.get_float_one_hot(u2)

                self.label_digit = self.label_digit.cuda()
                self.noise.data.normal_(0, 1)

                self.zcomp = Variable(z[self.j * self.batch_size: (self.j + 1) * self.batch_size], requires_grad=False)
                # Encoder
                self.fake_z = self.netE(self.input_var, self.label_digit)
                # random y + fake_z
                self.fake_img_fake_z = self.netG(self.one_hot_label_random, self.fake_z, self.c_code)

                # MineGAN
                self.fake_img = self.netG(self.one_hot_label, self.noise, self.c_code)

                # Update Discriminator networks
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i)
                    errD_total += errD

                self.train_BD()

                # Update the Generator networks
                errG_total, errG_ce, errG_ce_2, errG_Dmagn_fake, errG_Dmagn_fake_z_rep, \
                errG_Dy_fake, errG_Dy_fake_z_rep, errG_Dz = self.train_Gnet()
                errG_cycle_real = self.train_Gnet_cycle()
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # Update the Generator networks
                errC_total = self.train_Cnet()

                self.j += 1
                self.global_step += 1
                count = count + 1


                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    backup_para = copy_G_params(self.netG)
                    save_model(self.netM, self.netG, self.netE, avg_param_G, self.netC, self.netsD, count, self.model_dir)
                    # Save images
                    load_params(self.netG, avg_param_G)
                    self.netG.eval()
                    self.netC.eval()
                    self.netE.eval()
                    self.netM.eval()
                    self.netsD[0].eval()
                    with torch.set_grad_enabled(False):
                        fake_z = self.netE(self.input_var, self.u2_label_digit)
                        self.fake_imgfix_fake_z = self.netG(self.one_hot_label_random, fake_z, self.c_code)
                        self.fake_imgfix_fake_z_random = self.netG(self.one_hot_label, fake_z, self.c_code)
                        self.fake_imgfix_fake_z_mine  = self.netG(self.one_hot_label, fixed_noise, self.c_code)
                        fixed_noise_zcode_cycle = self.netE(self.fake_imgfix_fake_z_mine, self.label_digit)
                        self.fake_imgfix_fake_z_mine_cycle = self.netG(self.one_hot_label, fixed_noise_zcode_cycle, self.c_code)

                        compare.append(self.fake_imgfix_fake_z)
                        compare.append(self.fake_imgfix_fake_z_random)
                        compare.append(self.fake_imgfix_fake_z_mine)
                        compare.append(self.fake_imgfix_fake_z_mine_cycle)

                        acc, acc2 = self.calc_metrics_C(self.netC, self.netsD[0], self.test_data)
                        print(count)
                        print('Accuracy of the C on the %d test images: %.2f %% D_clas images: %.2f %%' % (
                        len(self.test_data) * cfg.TRAIN.BATCH_SIZE, acc, acc2))
                    save_img_results(self.imgs_tcpu, (compare),
                                     self.num_Ds, count, self.image_dir, self.summary_writer)
                    self.netC.train()
                    self.netG.train()
                    self.netE.train()
                    self.netM.train()
                    self.netsD[0].train()

                    load_params(self.netG, backup_para)

            alpha = 0.6
            Z = alpha * Z + (1. - alpha) * self.outputs
            z = Z * (1. / (1. - alpha ** (epoch + 1)))

            end_t = time.time()
            print('''[%d/%d][%d]
                          Loss_C: %.2f  Loss_G: %.2f Loss_D: %.2f 
                          errG_ce: %.2f, errG_ce_2: %.2f, errG_Dmagn_fake: %.2f, 
                          errG_Dmagn_fake_z_rep: %.2f, 
                          errG_Dy_fake: %.2f, errG_Dy_fake_z_rep: %.2f,
                          errG_cycle_real: %.6f 
                          Time: %.2fs
                      '''
                  % (epoch, self.max_epoch, self.num_batches,
                     errC_total.item(), errG_total.item(), errD_total.item(),
                     errG_ce.item(), errG_ce_2.item(), errG_Dmagn_fake.item(), errG_Dmagn_fake_z_rep.item(), \
                     errG_Dy_fake.item(), errG_Dy_fake_z_rep.item(), errG_cycle_real.item(),
                     end_t - start_t))


        save_model(self.netM, self.netG, self.netE, avg_param_G, self.netC, self.netsD, count, self.model_dir)

        self.summary_writer.close()


class SSCGAN_test(object):

    def __init__(self, dataloader, testloader):
        self.save_dir = os.path.join(cfg.SAVE_DIR, 'images')
        mkdir_p(self.save_dir)
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.dataloader = dataloader
        self.testloader = testloader

    def sample_pseudo_labels_onehot_1_label(self, num_classes, batch_size, choice):
        labels = np.random.choice(a=choice, size=batch_size, replace=False, p=None)
        pseudo_labels = torch.from_numpy(labels)
        pseudo_labels = pseudo_labels.type(torch.long).cuda()
        labels_onehot = np.eye(num_classes)[labels]
        pseudo_labels_onehot = torch.from_numpy(labels_onehot)
        pseudo_labels_onehot = pseudo_labels_onehot.type(torch.float).cuda()
        return pseudo_labels_onehot, pseudo_labels

    def sample_pseudo_labels_c_code(self, num_classes, batch_size, flag=False):
        labels = np.random.randint(low=0, high=num_classes, size=(batch_size))
        pseudo_labels = torch.from_numpy(labels)
        pseudo_labels = pseudo_labels.type(torch.long).cuda()
        labels_onehot = np.eye(num_classes)[labels]
        pseudo_labels_onehot = torch.from_numpy(labels_onehot)
        pseudo_labels_onehot = pseudo_labels_onehot.type(torch.float).cuda()
        return pseudo_labels_onehot, pseudo_labels

    def save_image(self, images, save_dir, iname):
        img_name = '%s.png' % (iname)
        full_path = os.path.join(save_dir, img_name)
        if (iname.find('mask') == -1) or (iname.find('foreground') != -1):
            img = images.add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(full_path)
        else:
            img = images.mul(255).clamp(0, 255).byte()
            ndarr = img.data.cpu().numpy()
            ndarr = np.reshape(ndarr, (ndarr.shape[-1], ndarr.shape[-1], 1))
            ndarr = np.repeat(ndarr, 3, axis=2)
            im = Image.fromarray(ndarr)
            im.save(full_path)

    def generate_definite_y_img(self):
        from models.model_sscgan import G_NET
        netG = G_NET().cuda().eval()
        model_dict = torch.load(cfg.TRAIN.NET_G, map_location='cuda:0')
        netG.load_state_dict(model_dict)
        print('Load', cfg.TRAIN.NET_G)
        nz = cfg.GAN.Z_DIM
        noise = torch.FloatTensor(self.batch_size, nz).cuda()
        cnt = 0
        # given number
        num = 123
        class_label = [num]
        for i in range(1000):
            noise.data.normal_(0, 1)
            y_code, y_code_digit = self.sample_pseudo_labels_onehot_1_label(cfg.CLASSES, 1, class_label)
            c_code, c_code_digit = self.sample_pseudo_labels_c_code(50, 1)
            with torch.set_grad_enabled(False):
                fake_img = netG(y_code, noise, c_code)
            self.save_image(fake_img[0], self.save_dir, 'fake_img_idx_' + str(cnt))
            cnt += 1
