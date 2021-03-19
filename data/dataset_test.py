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

    def __init__(self, phase,opt):
        super(DatasetTSMS, self).__init__()
        self.opt = opt
        self.phase = phase
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4

        self.paths_L, self.sizes_L = _get_paths_from_lmdb(opt['dataroot_L'])

        self.l_env = lmdb.open(opt['dataroot_L'], readonly=True, lock=False, readahead=False,
                                 meminit=False)


    def __getitem__(self, index):


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
        img_deblur = [img_L]#,img_h0]
        img_deblur = util.generate_pyramid(*img_deblur,n_scales=3)
        img_deblur =util.np2tensor(*img_deblur)
        img_L = img_deblur[0]
        
        return {'L0':img_L[0], 'ls':img_L, 'L': img_L, 'L_path':L_path}

    def __len__(self):
        return len(self.paths_L)

