import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from utils.fusion_loss import MaxGradLoss, MaxPixelLoss, PixelLoss, SSIM




class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.MaxGradLoss = MaxGradLoss(3)
        # when input img just one, PixelLoss=MaxPixelLoss
        self.PixelLoss = PixelLoss(3)
        self.MaxPixelLoss = MaxPixelLoss(1)
        self.msssim_loss = SSIM(window_size=24)

    def getPromptLoss(self, prompt):
        prompt_rgb, prompt_t = prompt

        prompt_loss = 0
        for i in range(len(prompt_t)):
            prompt_gt = torch.ones_like(prompt_t[i])  # B 1 H W
            prompt_loss += self.mse(prompt_rgb[i] + prompt_t[i], prompt_gt)

        return prompt_loss

    def forward(self, img_rgb, img_t, pred_img, prompt, epoch):
        # pred_img = self.unpatchify(pred)
        B, C, H, W = img_rgb.shape

        # pred_img = (pred_img) / 255
        # img_rgb = (img_rgb) / 255
        # img_t = (img_t) / 255
        prompt_loss = self.getPromptLoss(prompt)
        ssim_loss_rgb = 1 - self.msssim_loss(pred_img, img_rgb, normalize=True)
        ssim_loss_t = 1 - self.msssim_loss(pred_img, img_t, normalize=True)
        ssim_loss = (ssim_loss_rgb + ssim_loss_t) / 2
        if epoch < 20:
            pixel_loss = self.PixelLoss(pred_img, img_rgb, img_t)
            loss = pixel_loss + ssim_loss + prompt_loss

            # print(loss)
        else:
            max_grad_loss = self.MaxGradLoss(pred_img, img_rgb, img_t)
            max_pixel_loss = self.MaxPixelLoss(pred_img, img_rgb, img_t)
            loss = max_grad_loss + max_pixel_loss + 20 * prompt_loss + 0.5 * ssim_loss

        return loss, ssim_loss