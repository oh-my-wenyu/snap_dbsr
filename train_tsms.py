import os.path
import math
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


# --------------------------------------------
# github: https://github.com/cszn/KAIR
# --------------------------------------------


def main(json_path='options/train_tsms.json'):  

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G2')
    opt['path']['pretrained_netG2'] = init_path_G
    current_step = init_iter
    border = opt['scale']

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(phase, dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'val':
            val_set = define_Dataset(phase, dataset_opt)
            val_loader = DataLoader(val_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    model_2 = define_Model(opt, stage2=True)
    model_2.init_train()
    #logger.info(model_2.info_params())

#    #load model Deblur--------------------------------------------------------------
#    model_2_dict = model_2.netG.module.state_dict()
#    _, last_path_M1 = option.find_last_checkpoint(opt['path']['models'], net_type='G0')
#    model_1_saved = torch.load(last_path_M1)
#    model_1_dict = {k: v for k, v in model_1_saved.items() if k in model_2_dict.keys()}
#    model_2_dict.update(model_1_dict)
#    model_2.netG.module.load_state_dict(model_2_dict)
#
#    #load model SR----------------------------------------------------------------
#    model_12_dict = model_2.netG.module.state_dict()
#    _, last_path_M2 = option.find_last_checkpoint(opt['path']['models'], net_type='G1')
#    M2_saved = torch.load(last_path_M2)
#    M2_dict = {k: v for k, v in M2_saved.items() if k in model_12_dict.keys()}
#    model_12_dict.update(M2_dict)
#    model_2.netG.module.load_state_dict(model_12_dict)
#  


    # logger.info(model_2.info_network())
    #logger.info(model_2.info_params())

    for epoch in range(100000):  
        for i, train_data in enumerate(train_loader):

            current_step += 1

            model_2.update_learning_rate(current_step)

            model_2.feed_data(train_data)

            model_2.optimize_parameters(current_step)

            if current_step % opt['train']['checkpoint_save'] == 0:
                # logger.info('Saving the model.')
                model_2.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:
                # training info
                logs = model_2.current_log()  # such as loss
                message_tr = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                             model_2.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message_tr += '\t{:s}: {:.3e}'.format(k, v)

                avg_psnr = 0.0
                idx = 0

                for val_data in val_loader:
                    idx += 1
                    model_2.feed_data(val_data)
                    model_2.test()

                    visuals = model_2.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])
                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                # val log
                message_val = '\tTSMS Te_PSNR_avg: {:<.2f}dB'.format(avg_psnr)
                message = message_tr + message_val
                logger.info(message)
   
    logger.info('End of Two-Stage Multi-Scale training.')


if __name__ == '__main__':
    main()

