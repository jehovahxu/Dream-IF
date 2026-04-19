import os
import torch
import argparse
import datetime


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser = argparse.ArgumentParser(description="PyTorch")
        self.parser.add_argument('--data_root', default='', help="path to images")
        self.parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')
        self.parser.add_argument('--image_size', type=int, default=286, help='scale images to this size')
        # self.parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
        self.parser.add_argument('--bata', type=int, default=0.5, help='momentum parameters bata1')
        self.parser.add_argument('--batch_size', type=int, default=4,
                                 help='with batchSize=1 equivalent to instance normalization.')
        self.parser.add_argument('--num_works', type=int, default=16,
                                 help='with batchSize=1 equivalent to instance normalization.')
        self.parser.add_argument('--lr', type=float, default=0.0003)
        self.parser.add_argument('--val_every_epoch', type=int, default=5)
        self.parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
        self.parser.add_argument('--save_dir', type=str, default='./save_dir', help=' results are saved here')
        self.parser.add_argument('--resume', action='store_true', help=' results are saved here')
        self.parser.add_argument('--use_dp', action='store_true', help=' results are saved here')
        self.parser.add_argument('--resume_pth', type=str, default='')
        self.parser.add_argument('--save_name', type=str, default='first', help='the dir to save the training intermediate results')
        self.parser.add_argument('--checkpoints', type=str, default='./checkpoints', help=' models are saved here')
        self.parser.add_argument('--output', default='./output', help='folder to output images ')
        self.parser.add_argument('--datalist', default='files/list_train.txt', help='use a text to load dataset and you\
                                 also need switch list when you test')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # set_gpus
        str_ids = opt.gpu_id.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # only use one gpu
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        now = datetime.datetime.now().strftime("%y%m%d%H")
        cur_dir_name = now + '-' + opt.save_name
        cur_save_dir = os.path.join(opt.save_dir, cur_dir_name)
        opt.checkpoints = os.path.join(cur_save_dir, 'checkpoint')
        opt.samples = os.path.join(cur_save_dir, 'samples')
        opt.training_sample = os.path.join(opt.samples, 'training_sample')
        opt.output = os.path.join(cur_save_dir, 'last_result')

        mkdirs([opt.save_dir, opt.checkpoints, opt.samples, opt.training_sample, opt.output])
        # save to the disk
        opt.logging = os.path.join(cur_save_dir, 'logging.log')
        self.opt = opt
        return self.opt


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)