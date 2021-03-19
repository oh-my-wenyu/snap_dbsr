from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.nn.parallel import DataParallel  # , DistributedDataParallel

from models.select_network_stage import define_G
from models.model_base import ModelBase
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
import utils.utils_image as util


class ModelStage(ModelBase):
    """Train with pixel loss"""

    def __init__(self, opt, stage0=False, stage1=False, stage2=False):
        super(ModelStage, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.stage0 = stage0
        self.stage1 = stage1
        self.stage2 = stage2
        self.netG = define_G(opt,self.stage0,self.stage1,self.stage2).to(self.device)
        self.netG = DataParallel(self.netG)

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.opt_train = self.opt['train']  # training option
        self.load()  # load model
        self.netG.train()  # set training mode,for BN
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):

        if self.stage0:
            load_path_G = self.opt['path']['pretrained_netG0']
        elif self.stage1:
            load_path_G = self.opt['path']['pretrained_netG1']
        elif self.stage2:
            load_path_G = self.opt['path']['pretrained_netG2']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    # ----------------------------------------
    # save model
    # ----------------------------------------
    def save(self, iter_label):
        if self.stage0:
            self.save_network(self.save_dir, self.netG, 'G0', iter_label)
        elif self.stage1:
            self.save_network(self.save_dir, self.netG, 'G1', iter_label)
        elif self.stage2:
            self.save_network(self.save_dir, self.netG, 'G2', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data):
        if self.stage0:
            Ls = data['ls']
            self.Ls = util.tos(*Ls,device=self.device)
            Hs = data['hs']
            self.Hs = util.tos(*Hs,device=self.device)
        if self.stage1:
            self.L0 = data['L0'].to(self.device)
            self.H = data['H'].to(self.device)
        elif self.stage2:
            Ls = data['L']
            self.Ls = util.tos(*Ls,device=self.device)
            self.H = data['H'].to(self.device) #hide for test

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):

        self.G_optimizer.zero_grad()

        if self.stage0:
            self.Es = self.netG(self.Ls)
            _loss = []
            for (Es_i,Hs_i) in zip(self.Es,self.Hs):
                _loss += [self.G_lossfn(Es_i, Hs_i)]
            G_loss = sum(_loss)*self.G_lossfn_weight

        if self.stage1:
            self.E = self.netG(self.L0)
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)


        if self.stage2:
            self.E = self.netG(self.Ls)
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)


        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'],
                                           norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train[
            'G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train[
            'G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        if self.stage0:
            with torch.no_grad():
                self.Es = self.netG(self.Ls)
        elif self.stage1:
            with torch.no_grad():
                self.E = self.netG(self.L0)
        elif self.stage2:
            with torch.no_grad():
                self.E = self.netG(self.Ls)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self):
        out_dict = OrderedDict()
        if self.stage0:
            out_dict['L'] = self.Ls[0].detach()[0].float().cpu()
            out_dict['Es0'] = self.Es[0].detach()[0].float().cpu()
            out_dict['Hs0'] = self.Hs[0].detach()[0].float().cpu()
        elif self.stage1:
            out_dict['L'] = self.L0.detach()[0].float().cpu()
            out_dict['E'] = self.E.detach()[0].float().cpu()
            out_dict['H'] = self.H.detach()[0].float().cpu() #hide for test
 
        elif self.stage2:
            out_dict['L'] = self.Ls[0].detach()[0].float().cpu()
            out_dict['E'] = self.E.detach()[0].float().cpu()
            out_dict['H'] = self.H.detach()[0].float().cpu() #hide for test
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg



