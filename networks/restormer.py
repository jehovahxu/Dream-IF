## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import cv2
import numpy as np

from einops import rearrange
from util.fusion_loss import *
from util.mefssim import MEF_MSSSIM

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.restoration_3 = Restoration_all(dim_in=int(dim*2**2), dim_out=int(dim*2**2), heads=heads,
                                          prompt_dim=320, prompt_len=5, prompt_size=64, lin_dim=dim * 2 ** 2)
        self.restoration_2 = Restoration_all(dim_in=int(dim * 2 ** 1), dim_out=int(dim * 2 ** 1), heads=heads,
                                          prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=dim * 2 ** 1)
        self.restoration_1 = Restoration_all(dim_in=int(dim * 2 ** 0), dim_out=int(dim * 2 ** 1), heads=heads,
                                          prompt_dim=64, prompt_len=5, prompt_size=16, lin_dim=dim * 2 ** 0)
        # self.restoration_1 = Restoration_all(dim_in=int(dim * 2 ** 0), dim_out=int(dim * 2 ** 0), heads=heads,
        #                                   prompt_dim=64, prompt_len=5, prompt_size=8, lin_dim=dim * 2 ** 0)

        self.mse = torch.nn.MSELoss()
        self.MaxGradLoss = MaxGradLoss(3)
        # when input img just one, PixelLoss=MaxPixelLoss
        self.PixelLoss = PixelLoss(3)
        self.MaxPixelLoss = MaxPixelLoss(1)
        self.MFFselect = MaxGradTokenSelect()
        self.msssim_loss = SSIM(window_size=24)
        self.MEFSSIM = MEF_MSSSIM(is_lum=True)
        self.warmup_epochs = 20


    def forward_net(self, img1, img2):
        prompt_x_list, prompt_t_list = [], []

        inp_enc_level1_a = self.patch_embed(img1)
        inp_enc_level1_b = self.patch_embed(img2)
        out_enc_level1_a = self.encoder_level1(inp_enc_level1_a)
        out_enc_level1_b = self.encoder_level1(inp_enc_level1_b)

        inp_enc_level2_a = self.down1_2(out_enc_level1_a)
        out_enc_level2_a = self.encoder_level2(inp_enc_level2_a)

        inp_enc_level2_b = self.down1_2(out_enc_level1_b)
        out_enc_level2_b = self.encoder_level2(inp_enc_level2_b)

        inp_enc_level3_a = self.down2_3(out_enc_level2_a)
        out_enc_level3_a = self.encoder_level3(inp_enc_level3_a)

        inp_enc_level3_b = self.down2_3(out_enc_level2_b)
        out_enc_level3_b = self.encoder_level3(inp_enc_level3_b)

        inp_enc_level4_a = self.down3_4(out_enc_level3_a)
        latent_a = self.latent(inp_enc_level4_a)
        inp_enc_level4_b = self.down3_4(out_enc_level3_b)
        latent_b = self.latent(inp_enc_level4_b)


        inp_dec_level3_a = self.up4_3(latent_a)
        inp_dec_level3_b = self.up4_3(latent_b)


        # inp_dec_level3 = torch.cat([inp_dec_level3_a, out_enc_level3], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        prompt_x, prompt_t, inp_dec_level3_a, inp_dec_level3_b = self.restoration_3(inp_dec_level3_a, inp_dec_level3_b)
        prompt_x_list.append(prompt_x)
        prompt_t_list.append(prompt_t)

        out_dec_level3_a = self.decoder_level3(inp_dec_level3_a)
        out_dec_level3_b = self.decoder_level3(inp_dec_level3_b)

        # inp_dec_level2 = self.up3_2(out_dec_level3)
        # inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2_a = self.up3_2(out_dec_level3_a)
        inp_dec_level2_b = self.up3_2(out_dec_level3_b)

        prompt_x, prompt_t, inp_dec_level2_a, inp_dec_level2_b = self.restoration_2(inp_dec_level2_a, inp_dec_level2_b)
        prompt_x_list.append(prompt_x)
        prompt_t_list.append(prompt_t)

        # inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2_a = self.decoder_level2(inp_dec_level2_a)
        out_dec_level2_b = self.decoder_level2(inp_dec_level2_b)


        inp_dec_level1_a = self.up2_1(out_dec_level2_a)
        inp_dec_level1_b = self.up2_1(out_dec_level2_b)

        # inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        prompt_x, prompt_t, inp_dec_level1_a, inp_dec_level1_b = self.restoration_1(inp_dec_level1_a, inp_dec_level1_b)
        prompt_x_list.append(prompt_x)
        prompt_t_list.append(prompt_t)

        out_dec_level1_a = self.decoder_level1(inp_dec_level1_a)
        out_dec_level1_b = self.decoder_level1(inp_dec_level1_b)

        out_dec_level1 = out_dec_level1_a + out_dec_level1_b
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)         ## del inp_img

        # #### For Dual-Pixel Defocus Deblurring Task ####
        # if self.dual_pixel_task:
        #     out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
        #     out_dec_level1 = self.output(out_dec_level1)
        # ###########################
        # else:
        #     out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1, prompt_x_list, prompt_t_list

    def getPromptLoss(self, prompt):
        prompt_rgb, prompt_t = prompt

        prompt_loss = 0
        for i in range(len(prompt_t)):
            prompt_gt = torch.ones_like(prompt_t[i])  # B 1 H W
            prompt_loss += self.mse(prompt_rgb[i] + prompt_t[i], prompt_gt)

        return prompt_loss

    def forward_loss_taskRGBT(self, img_rgb, img_t, pred_img, prompt, epoch):
        # pred_img = self.unpatchify(pred)
        B, C, H, W = img_rgb.shape
        # if self.upsample:
        #     pred_img = F.interpolate(pred, size=(H, W))
        # import pdb;pdb.set_trace()
        # pred_img = recover_Norm(pred_img) / 255
        # img_rgb = recover_Norm(img_rgb) / 255
        # img_t = recover_Norm(img_t) / 255

        # prompt B,N,C
        prompt_loss = self.getPromptLoss(prompt)

        ssim_loss_rgb = 1 - self.msssim_loss(pred_img, img_rgb, normalize=True)
        ssim_loss_t = 1 - self.msssim_loss(pred_img, img_t, normalize=True)
        ssim_loss = (ssim_loss_rgb + ssim_loss_t) / 2

        if epoch < self.warmup_epochs:
            pixel_loss = self.PixelLoss(pred_img, img_rgb, img_t)
            loss = pixel_loss + 20 * prompt_loss + ssim_loss
            loss = {
                "loss": loss,
                "pixel_loss": pixel_loss,
                "prompt_loss": prompt_loss,
            }
            # print(loss)
        else:
            max_grad_loss = self.MaxGradLoss(pred_img, img_rgb, img_t)
            max_pixel_loss = self.MaxPixelLoss(pred_img, img_rgb, img_t)
            loss = max_grad_loss + max_pixel_loss + 20 * prompt_loss + 0.5 * ssim_loss
            loss = {"loss": loss,
                    "ssim_loss": ssim_loss,
                    "max_grad_loss": max_grad_loss,
                    "max_pixel_loss": max_pixel_loss,
                    "prompt_loss": prompt_loss,
                    }

        return loss, pred_img, prompt

    def forward_loss_split(self, img_rgb, img_t, pred, prompt, epoch):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        prompt: [# B 1 H W],
        """
        loss, pred_img, prompt_tuple = self.forward_loss_taskRGBT(img_rgb, img_t, pred, prompt, epoch)
        return loss, pred_img, prompt_tuple

    def forward(self, img_t, img_rgb, epoch):
        if len(img_rgb.shape) == 5:
            img_rgb = img_rgb.squeeze()
            img_t = img_t.squeeze()
        # latent, t, prompt_rgb, prompt_t, encoder_info = self.forward_encoder(img_rgb, img_t)
        pred, de_prompt_rgb, de_prompt_t = self.forward_net(img_rgb, img_t)  # [N, L, p*p*3]
        with torch.cuda.amp.autocast(enabled=False):
            loss, pred, prompt_tuple = self.forward_loss_split(img_rgb, img_t, pred,
                                                               (de_prompt_rgb, de_prompt_t),
                                                                 epoch)
        return loss, pred, prompt_tuple



def draw_features(x, savename='', width=640, height=512):
    img = torch.mean(x, 1).squeeze().cpu().numpy()
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 1e-6))

    flat_img = img.flatten()
    n_elements = flat_img.size

    # 计算前 10% 和后 10% 的边界索引
    lower_bound = int(0.05 * n_elements)
    upper_bound = int(0.95 * n_elements)

    # 对一维数组进行排序以找到前 10% 和后 10% 的位置
    sorted_indices = np.argsort(flat_img)
    flat_img[sorted_indices[:lower_bound]] = 0  # 前 10% 设置为 0
    flat_img[sorted_indices[upper_bound:]] = 1  # 后 10% 设置为 1

    # 将一维数组重塑回原始形状
    # img = flat_img.reshape(img.shape)

    # 将图像值乘以 255MEF
    img = (img * 255).astype(np.uint8)

    img = img.astype(np.uint8)  # 转成unit8
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
    # att_tuple = [cv2.applyColorMap(att.cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET) for att in att_tuple_gray]
    # img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (width, height))
    cv2.imwrite(savename, img)

def draw_features_list(x_A, x_B, imp_A, imp_B, refine_A, refine_B, level=-1):
    if level == -1:
        level_dict = {384: '1', 192: '2', 96: '3', 48: '4'}
        level = level_dict[x_A.shape[1]]
    draw_features(512, 512, x_A, 'feature_vis/f%s_x_A.png' % level)
    draw_features(512, 512, x_B, 'feature_vis/f%s_x_B.png' % level)
    draw_features(512, 512, imp_A, 'feature_vis/f%s_imp_A.png' % level)
    draw_features(512, 512, imp_B, 'feature_vis/f%s_imp_B.png' % level)
    draw_features(512, 512, refine_A, 'feature_vis/f%s_refine_A.png' % level)
    draw_features(512, 512, refine_B, 'feature_vis/f%s_refine_B.png' % level)


class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class PromptBlock(nn.Module):
    def __init__(self, dim_in, dim_out, heads, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptBlock, self).__init__()
        self.noise_level = TransformerBlock(dim=dim_in + prompt_dim, num_heads=heads[2])
        self.reduce_noise_level = nn.Conv2d(dim_in, dim_out, kernel_size=1)

        self.PGM = PromptGenBlock(prompt_dim, prompt_len, prompt_size, lin_dim)
        # self.noise_level = TransformerBlock(dim=dim_in+prompt_dim, num_heads=heads[2])
        self.reduce_noise_level = nn.Conv2d(dim_in + prompt_dim, dim_out, kernel_size=1)

    def forward(self, x, prompt):
        out_dec = self.PGM(torch.cat([x, prompt], 1))
        out_dec_level = torch.cat([x, out_dec], 1)
        out_dec_level = self.noise_level(out_dec_level)
        out_dec_level = self.reduce_noise_level(out_dec_level)

        return out_dec_level

class Restoration(nn.Module):
    def __init__(self, dim_in, dim_out, heads, prompt_dim=128,
                 prompt_len=10, prompt_size=96, lin_dim=192):
        super(Restoration, self).__init__()
        # self.PGM = PromptGenBlock(prompt_dim, prompt_len, prompt_size, lin_dim)
        self.noise_level = TransformerBlock(dim=dim_in+prompt_dim, num_heads=heads[2])
        self.reduce_noise_level = nn.Conv2d(dim_in+prompt_dim, dim_out, kernel_size=1)
        self.Conv3_1 = nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1)
        self.Conv3_2 = nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1)

        self.Conv3_3 = nn.Conv2d(dim_in*2, dim_out, kernel_size=3, stride=1, padding=1)
        self.PB = PromptBlock(dim_in, dim_out, heads, prompt_dim,
                 prompt_len, prompt_size, lin_dim + 1)

        self.norm = LayerNorm(dim_in*2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, relative_x, relative_y):

        x_1 = self.Conv3_1(x)
        x_2 = self.Conv3_1(x)

        y_1 = self.Conv3_2(y)

        x_cross = x_1 + y_1 * relative_y

        x_self = x_2 + self.PB(x, relative_x)

        x = self.Conv3_3(torch.cat([x_cross, x_self], 1))

        # x = , prompt], 1))
        return x

class Restoration_all(nn.Module):
    def __init__(self, dim_in, dim_out, heads, prompt_dim=128,
                 prompt_len=5, prompt_size=96, lin_dim=192):
        super(Restoration_all, self).__init__()
        self.restoration_A = Restoration(dim_in, dim_out, heads, prompt_dim, prompt_len, prompt_size, lin_dim)
        self.restoration_B = Restoration(dim_in, dim_out, heads, prompt_dim, prompt_len, prompt_size, lin_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.agv_pool = nn.AdaptiveAvgPool2d(1)

        self.norm1 = nn.BatchNorm2d(dim_in)
        self.norm2 = nn.BatchNorm2d(dim_in)
        self.conv = nn.Conv2d(dim_in * 2, dim_out*2, kernel_size=3, stride=1, padding=1)

    def forward(self, x_A, x_B):
        # x_A = self.norm1(x_A)
        # x_B = self.norm1(x_B)
        y = self.conv(torch.cat([x_A, x_B], 1))
        x_A, x_B = torch.chunk(y, 2, dim=1)
        prompt_x = torch.mean(x_A, 1, keepdim=True)
        # prompt_x = self.agv_pool(x_A)
        prompt_x = self.sigmoid(prompt_x)
        prompt_t = torch.mean(x_B, 1, keepdim=True)
        # prompt_t = self.agv_pool(x_B)
        prompt_t = self.sigmoid(prompt_t)
        # refine_A = self.restoration_A(x_A, x_B, prompt_x, prompt_t)
        # refine_B = self.restoration_B(x_B, x_A, prompt_t, prompt_x)
        refine_A, refine_B = x_A, x_B
        refine_A = prompt_x * refine_A
        refine_B = prompt_t * refine_B

        # draw_features_list(x_A, x_B, imp_A, imp_B, refine_A, refine_B)
        return prompt_x, prompt_t, refine_A, refine_B



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features * 2, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    # def forward(self, x):
    #     x = self.project_in(x)
    #     x = self.dwconv(x)
    #     x = F.gelu(x)
    #     x = self.project_out(x)
    #     return x

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

def init_model():
    # models_mae = MaskedAutoencoderViT()
    print("Model Type: Restormer")
    # Initialize the model and load the parameters
    model = Restormer()
    # model = load_pretrained_weights(model, weights_path='/root/nas-public-linkdata/dataset/other/gaussian_color_denoising_blind.pth')
    # model = mae_vit_large_patch16(config=None)
    # model = load_pretrained_weights(model, weights_path='/root/nas-public-linkdata/dataset/other/mae_visualize_vit_large_ganloss.pth')

    return model
def load_pretrained_weights(
        model,
        model_name=None,
        weights_path=None,

        strict=False,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    print("load_model_info:", model_name, weights_path)

    # Load weights
    state_dict = torch.load(weights_path)
    # state_dict = state_dict

    # Modifications to load partial state dict
    # expected_missing_keys = []
    # for key in expected_missing_keys:
    #     state_dict.pop(key)

    ret = model.load_state_dict(state_dict, strict=False)
    for name, parms in model.named_parameters():
        if name in state_dict:
            parms.requires_grad = False

    return model

if __name__ == '__main__':
    input_A = torch.randn((1, 3, 96, 96))
    input_B = torch.randn((1, 3, 96, 96))
    model = Restormer()
    output, _, _ = model(input_A, input_B)
    print(output.shape)

