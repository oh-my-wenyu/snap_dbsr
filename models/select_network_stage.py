import functools
import torch
from torch.nn import init




# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt,stage0=False,stage1=False,stage2=False):
    opt_net = opt['netG']
    if stage0:
        net_type = opt_net['net_type0']
    elif stage1:
        net_type = opt_net['net_type1']
    elif stage2:
        net_type = opt_net['net_type2']
    
    
    # ----------------------------------------
    # deblur 
    # ----------------------------------------
    if net_type == 'deblur':  # multiscale
        from models.network_deblur import DeblurNet as net
        netG = net(n_feats=opt_net['n_feats'],
                   n_blocks=opt_net['n_blocks'],
                   kernel_size=opt_net['kernel_size'],
                   n_scales=opt_net['n_scales'])

    # ----------------------------------------
    # se
    # ----------------------------------------
    elif net_type == 'sr':  # multiscale
        from models.network_sr import SRNet as net
        netG = net(dn_feats=opt_net['dn_feats'],
                   dd_feats=opt_net['dd_feats'],
                   dn_blocks=opt_net['dn_blocks'],
                   n_feats=opt_net['n_feats'],
                   n_blocks=opt_net['n_blocks'],
                   kernel_size=opt_net['kernel_size'])



    # ---------------------------------------- 
    # tsms
    # ---------------------------------------- 
    elif net_type == 'tsms':
        from models.network_tsms import TSMSNet as net
        netG = net(n_feats_1=opt_net['n_feats_1'],
                   n_blocks_1=opt_net['n_blocks_1'],
                   kernel_size_1=opt_net['kernel_size_1'],
                   n_scales_1=opt_net['n_scales_1'],
                   dn_feats_2=opt_net['dn_feats_2'],
                   dd_feats_2=opt_net['dd_feats_2'],
                   dn_blocks_2=opt_net['dn_blocks_2'],
                   n_feats_2=opt_net['n_feats_2'],
                   n_blocks_2=opt_net['n_blocks_2'],
                   kernel_size_2=opt_net['kernel_size_2'])


    # ----------------------------------------
    # others
    # ----------------------------------------
    # TODO

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG



"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """
    print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
    net.apply(fn)




