import pickle
import os
import lmdb
import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
from skimage.transform import pyramid_gaussian


def _get_paths_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'),
                                 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes

def _read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img

class DatasetTSMS(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, phase,opt):
        super(DatasetTSMS, self).__init__()
        self.opt = opt
        self.phase = phase
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H, self.sizes_H = _get_paths_from_lmdb(opt['dataroot_H'])
        self.paths_L, self.sizes_L = _get_paths_from_lmdb(opt['dataroot_L'])
        if self.phase == 'val':
            self.paths_H, self.sizes_H = self.paths_H[9::10], self.sizes_H[9::10]
            self.paths_L, self.sizes_L = self.paths_L[9::10], self.sizes_L[9::10]


        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

        self.h_env = lmdb.open(opt['dataroot_H'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.l_env = lmdb.open(opt['dataroot_L'], readonly=True, lock=False, readahead=False,
                                 meminit=False)


    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        H_resolution = [int(s) for s in self.sizes_H[index].split('_')]
        img_H = _read_img_lmdb(self.h_env, H_path, H_resolution)
        img_H = util.uint2single(img_H)
        img_H = util.modcrop(img_H, self.sf)

        # ----------------------------------------------
        # get down4 sharp image(img_h0) to be target in train for deblur
        # ----------------------------------------------
        img_h0 = util.imresize_np(img_H, 1 / 4, True)

        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        L_resolution = [int(s) for s in self.sizes_L[index].split('_')]
        img_L = _read_img_lmdb(self.l_env, L_path, L_resolution)
        img_L = util.uint2single(img_L)

#        # ----------------------------------------------
#        # get blur image to be input in train for deblur
#        # ----------------------------------------------
#        img_l0 = img_L

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape
            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
            img_h0 = img_h0[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            img_L, img_H, img_h0 = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode), util.augment_img(img_h0, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_deblur = [img_L,img_h0]
        img_deblur = util.generate_pyramid(*img_deblur,n_scales=3)
        img_deblur =util.np2tensor(*img_deblur)
        img_ls = img_deblur[0]
        img_hs = img_deblur[1]
        img_L = img_deblur[0]
        img_H = util.single2tensor3(img_H) 
        img_L0 = img_L[0]
        
        #print(img_L[0].shape,img_H.shape,img_ls[0].shape,img_hs[0].shape)
        return {'L0':img_L0,'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path, 'ls': img_ls, 'hs': img_hs}

    def __len__(self):
        return len(self.paths_H)

