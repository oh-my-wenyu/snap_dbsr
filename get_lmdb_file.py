import glob
import os
import pickle
import sys

import cv2
import lmdb
import numpy as np
from tqdm import tqdm
import utils.utils_image as util

def main(mode):
    lmdb_path = 'datasets/val_reds_sharp.lmdb'
    data_path = 'NTIREData/val/val_sharp'#train_blur_bicubic train_sharp

    if mode == 'creating':
        opt = {
            'name': 'train_sharp',
            'img_folder': data_path,
            'lmdb_save_path': lmdb_path,
            'commit_interval': 500, 
            'num_workers': 16,
        }
        general_image_folder(opt)
    elif mode == 'testing':
        test_lmdb(lmdb_path, index=5)


def general_image_folder(opt):
therwise, it will store every resolution info.

    img_folder = opt['img_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}

    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    # read all the image paths to a list

    print('Reading image path list ...')
    all_img_list = util.get_image_paths(img_folder)
    keys = []
    for img_path in all_img_list:
        img_path_split = img_path.split('/')[-2:]
        img_name_ext = img_path_split[0] + '_' + img_path_split[1]
        img_name, ext = os.path.splitext(img_name_ext)
        keys.append(img_name)

    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    txn = env.begin(write=True)
    resolutions = []
    tqdm_iter = tqdm(enumerate(zip(all_img_list, keys)), total=len(all_img_list), leave=False)
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))

        key_byte = key.encode('ascii')
        data = util.imread_uint(path, 3)
        H, W, C = data.shape
        resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))

        txn.put(key_byte, data)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    assert len(keys) == len(resolutions)
    if len(set(resolutions)) <= 1:
        meta_info['resolution'] = [resolutions[0]]
        meta_info['keys'] = keys
        print('All images have the same resolution. Simplify the meta info.')
    else:
        meta_info['resolution'] = resolutions
        meta_info['keys'] = keys
        print('Not all images have the same resolution. Save meta info for each image.')

    pickle.dump(meta_info, open(os.path.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def test_lmdb(dataroot, index=1):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))

    # read one image
    key = meta_info['keys'][index]
    print('Reading {} for test.'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    print(img_flat.shape)
    C, H, W = [int(s) for s in meta_info['resolution'][0].split('_')]
    img = img_flat.reshape(H, W, C)
    #cv2.namedWindow('Test')
    #cv2.imshow('Test', img)
    #cv2.waitKeyEx()
    cv2.imwrite('lmdb_test.png',img[:,:,[2,1,0]])


if __name__ == "__main__":
    #mode = creating or testing
    main(mode='testing')
    #main(mode='creating')




