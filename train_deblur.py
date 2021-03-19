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


def main(json_path='options/train_deblur.json'):  

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
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G0')
    opt['path']['pretrained_netG0'] = init_path_G
    current_step = init_iter


    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

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

    '''
    # ----------------------------------------
    # Step--3 (model_1)
    # ----------------------------------------
    '''

    model_1 = define_Model(opt,stage0=True)
    #logger.info(model_1.info_network())
    model_1.init_train()
    #logger.info(model_1.info_params())

    for epoch in range(10000):
        for i, train_data in enumerate(train_loader):

            current_step += 1

            model_1.update_learning_rate(current_step)

            model_1.feed_data(train_data)

            model_1.optimize_parameters(current_step)

            if current_step % opt['train']['checkpoint_save'] == 0:
                # logger.info('Saving the model.')
                model_1.save(current_step)

            # -------------------------------
            # model_1 testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:
                # training info
                logs = model_1.current_log()  # such as loss
                message_tr = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                             model_1.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message_tr += '\t{:s}: {:.3e}'.format(k, v)

                avg_psnr = 0.0
                idx = 0

                for val_data in val_loader:
                    idx += 1

                    model_1.feed_data(val_data)
                    model_1.test()

                    visuals = model_1.current_visuals()
                    E_img = util.tensor2uint(visuals['Es0'])
                    H_img = util.tensor2uint(visuals['Hs0'])
                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    avg_psnr += current_psnr
                

                avg_psnr = avg_psnr / idx

                # testing log
                message_val = '\tStage Deblur Val_PSNR_avg: {:<.2f}dB'.format(avg_psnr)
                message = message_tr + message_val
                logger.info(message)

    logger.info('End of Stage Deblur training.')
    

if __name__ == '__main__':
    main()

