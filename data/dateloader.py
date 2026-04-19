from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import os
from glob import glob
from utils import utils_blindsr as ubsr
from utils import utils_image
import cv2
import numpy as np
import transforms as T

class CustomDataset(Dataset):
    def __init__(self, data_root, phase="train"):
        self.phase = phase
        if phase == 'train':
            self.transform = T.Compose([
                                        T.RandomCrop(96),
                                        T.RandomHorizontalFlip(0.5),
                                        T.RandomVerticalFlip(0.5),
                                        # T.RandomMask(0.1),
                                        T.ToTensor()])
            self.infrared_path = glob(os.path.join(data_root, 'infrared/train/*'))
            self.visible_path = os.path.join(data_root, 'visible/train/')

        else:
            self.transform = T.Compose([T.Resize_16(),
                                        T.ToTensor()])
            self.infrared_path = glob(os.path.join('/root/nas-public-linkdata/dataset/imagefusion/LLVIP_tiny_New/', 'ir/*'))
            self.visible_path = os.path.join('/root/nas-public-linkdata/dataset/imagefusion/LLVIP_tiny_New/', 'vi')

        self.sf = 1
        self.sigma = 50

    def __len__(self):
        return len(self.infrared_path)

    def __getitem__(self, item):
        image_A_path = self.infrared_path[item]
        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]
        image_B_path = os.path.join(self.visible_path, image_A_path.split('/')[-1])
        # image_A = Image.open(image_A_path).convert(mode='RGB')
        # image_B = Image.open(image_B_path).convert(mode='RGB')

        image_A = utils_image.imread_uint(image_A_path, 3)
        image_A = utils_image.uint2single(image_A)

        image_B = utils_image.imread_uint(image_B_path, 3)
        image_B = utils_image.uint2single(image_B)

        # print(image_A.size)
        if self.phase == 'train':
            lq_patchsize = (image_A.shape[0], image_A.shape[1])

            # if np.random.randn() > 0.8:
            #     # image_lq_A, image_gt_A = ubsr.degradation_bsrgan(image_A, sf=self.sf, lq_patchsize=lq_patchsize)
            #     noise = np.random.normal(0, self.sigma / 255, image_A.shape)
            #     image_gt_A = image_A
            #     image_lq_A = image_A + noise
            # else:
            #     image_lq_A, image_gt_A = image_A, image_A
            # if np.random.randn() > 0.8:
            #
            #     noise = np.random.normal(0, self.sigma / 255, image_A.shape)
            #     image_gt_B = image_B
            #     image_lq_B = image_B + noise
            #     # image_lq_B, image_gt_B = ubsr.degradation_bsrgan(image_B, sf=self.sf, lq_patchsize=lq_patchsize)
            # else:
            #     image_lq_B, image_gt_B = image_B, image_B
            image_lq_A, image_gt_A = image_A, image_A
            image_lq_B, image_gt_B = image_B, image_B
        else:
            image_lq_A, image_gt_A = image_A, image_A
            image_lq_B, image_gt_B = image_B, image_B

        # image_A = cv2.resize(utils_image.single2uint(image_lq_A), (int(self.sf * image_lq_A.shape[1]), int(self.sf * img_lq.shape[0])),
        #                         interpolation=0)
        # image_B = cv2.resize(utils_image.single2uint(image_lq_B), (int(self.sf * image_lq_B.shape[1]), int(sf * img_lq.shape[0])),
        #                         interpolation=0)
        # Apply any specified transformations
        # import pdb;pdb.set_trace()
        # print(image_lq_A.transpose(2,0,1).shape)
        image_lq_A = Image.fromarray((image_lq_A.clip(0,1)*255).astype(np.uint8))
        image_lq_B = Image.fromarray((image_lq_B.clip(0,1)*255).astype(np.uint8))

        image_gt_A = Image.fromarray((image_gt_A.clip(0,1)*255).astype(np.uint8))
        image_gt_B = Image.fromarray((image_gt_B.clip(0,1)*255).astype(np.uint8))

        if self.transform is not None:
            image_lq_A, image_gt_A, image_lq_B, image_gt_B = self.transform(image_lq_A, image_gt_A, image_lq_B, image_gt_B)

        return image_lq_A, image_gt_A, image_lq_B, image_gt_B, {'name':name}

    @staticmethod
    def collate_fn(batch):
        image_lq_A, image_gt_A, image_lq_B, image_gt_B, name = zip(*batch)
        image_lq_A = torch.stack(image_lq_A, dim=0)
        image_gt_A = torch.stack(image_gt_A, dim=0)
        image_lq_B = torch.stack(image_lq_B, dim=0)
        image_gt_B = torch.stack(image_gt_B, dim=0)
        return image_lq_A, image_gt_A, image_lq_B, image_gt_B, name