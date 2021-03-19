import os.path
import math
import cv2
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model_stage import define_Model

def main(json_path='options/val_tsms.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)

    logger_name = 'val_msmd_patch'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    for phase, dataset_opt in opt['datasets'].items():
        test_set = define_Dataset(phase,dataset_opt)
        test_loader = DataLoader(test_set, batch_size=1,
                                 shuffle=False, num_workers=1,
                                 drop_last=False, pin_memory=True)

    model = define_Model(opt,stage2=True)
    model.load()
    avg_psnr = 0.0
    idx = 0

    for test_data in test_loader:
        idx += 1
        image_name = os.path.basename(test_data['L_path'][0])
        image_name = image_name +'.png'
        save_img_path = os.path.join(opt['path']['images'],image_name)

        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        E_img = util.tensor2uint(visuals['E'])
        #print(E_img.shape)
        H_img = util.tensor2uint(visuals['H'])

        # -----------------------
        # save estimated image E
        # -----------------------
        util.imsave(E_img, save_img_path)
        # -----------------------
        # calculate PSNR
        # -----------------------
        current_psnr = util.calculate_psnr(E_img, H_img, border=4)
        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name, current_psnr))

        avg_psnr += current_psnr

    avg_psnr = avg_psnr / idx

    # testing log
    message_te = '\tVal_PSNR_avg: {:<.2f}dB'.format(avg_psnr)
    logger.info(message_te)


if __name__ == '__main__':
    main()

