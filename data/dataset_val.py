import pickle
import os
import lmdb
import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
from skimage.transform import pyramid_gaussian


def _get_paths_from_lmdb(dataroot):

    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'),
                                 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes

def _read_img_lmdb(env, key, size):
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img

class DatasetTSMS(data.Dataset):

    def __init__(self, phase,opt):
        super(DatasetTSMS, self).__init__()
        self.opt = opt
        self.phase = phase
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4

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


        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_deblur = [img_L,img_h0]
        img_deblur = util.generate_pyramid(*img_deblur,n_scales=3)
        img_deblur =util.np2tensor(*img_deblur)
        img_L = img_deblur[0]
        img_H = util.single2tensor3(img_H) 
        img_L0 = img_L[0]
        
        return {'L0':img_L0, 'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path, 'ls': img_L, 'hs': img_deblur[1]}

    def __len__(self):
        return len(self.paths_H)

