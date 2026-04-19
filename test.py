import os
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision.transforms import functional as F
from networks.restormer import Restormer
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def draw_features(x, savename='', width=640, height=512):
    img = torch.mean(x, 1).squeeze().cpu().numpy()

    flat_img = img.flatten()
    n_elements = flat_img.size

    lower_bound = int(0.10 * n_elements)
    upper_bound = int(0.90 * n_elements)

    # 对一维数组进行排序以找到前 10% 和后 10% 的位置
    sorted_indices = np.argsort(flat_img)
    flat_img[sorted_indices[:lower_bound]] = flat_img[sorted_indices[lower_bound]]  # 前 10% 设置为 0
    flat_img[sorted_indices[upper_bound:]] = flat_img[sorted_indices[upper_bound]]  # 后 10% 设置为 1

    pmin = np.min(flat_img)
    pmax = np.max(flat_img)
    flat_img = ((flat_img - pmin) / (pmax - pmin + 1e-6))

    img = flat_img.reshape(img.shape)

    img = (img * 255).astype(np.uint8)

    img = img.astype(np.uint8)  # 转成unit8
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (width, height))
    cv2.imwrite(savename, img)




def main(args):
    root_path = args.dataset_path
    save_path = args.save_path
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    save_att_path = os.path.join(save_path, 'attn')
    os.makedirs(save_att_path, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(int(args.gpu_id))
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']


    ### clean
    visible_root = os.path.join(args.dataset_path, "vi")
    infrared_root = os.path.join(args.dataset_path, "ir")

    visible_path = [os.path.join(visible_root, i) for i in os.listdir(visible_root)
                  if os.path.splitext(i)[-1] in supported]
    infrared_path = [os.path.join(infrared_root, i) for i in os.listdir(infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    visible_path.sort()
    infrared_path.sort()

    print("Find the number of visible image: {},  the number of the infrared image: {}".format(len(visible_path), len(infrared_path)))
    assert len(visible_path) == len(infrared_path), "The number of the source images does not match!"

    print("Begin to run!")
    # import pdb;pdb.set_trace()
    with torch.no_grad():
        model = Restormer().to(device)

        model_weight_path = args.weights_path
        model_weight = torch.load(model_weight_path, map_location=device)
        print(model_weight['epoch'])
        model.load_state_dict(model_weight['model'])

        model.eval()

    for i in range(len(visible_path)):
        ir_path = infrared_path[i]
        vi_path = visible_path[i]

        img_name = vi_path.replace("\\", "/").split("/")[-1]
        assert os.path.exists(ir_path), "file: '{}' dose not exist.".format(ir_path)
        assert os.path.exists(vi_path), "file: '{}' dose not exist.".format(vi_path)

        ir = Image.open(ir_path).convert(mode="RGB")
        vi = Image.open(vi_path).convert(mode="RGB")
        vi_array = np.array(vi)
        height, width = vi.size
        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        ir = ir.resize((new_height, new_width))
        vi = vi.resize((new_height, new_width))

        ir = F.to_tensor(ir)
        vi = F.to_tensor(vi)

        ir = ir.unsqueeze(0).cuda()
        vi = vi.unsqueeze(0).cuda()
        with torch.no_grad():
            loss_dict, I_fused, att_tuple = model(ir, vi, 0)
            prompt_rgb, prompt_t = att_tuple
            fused_img = tensor2numpy(I_fused)
            draw_features(prompt_rgb[-2], os.path.join(save_att_path, 'prompt_rgb_{}.png'.format(img_name)))
            draw_features(prompt_t[-2], os.path.join(save_att_path, 'prompt_t_{}.png'.format(img_name)))
            # fused_img = mergy_Y_RGB_to_YCbCr(i, vi)
            save_pic(fused_img, save_path, img_name)

        print("Save the {}".format(img_name))
    print("Finish! The results are saved in {}.".format(save_path))

def tensor2numpy(img_tensor):
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img

def mergy_Y_RGB_to_YCbCr(img1, img2):
    img1 = img1.squeeze(0).cpu().numpy()
    img1 = np.transpose(img1, [1, 2, 0])
    img1_YCbCr = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
    Y_channel = img1_YCbCr[:, :, :1]

    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])
    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img


def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic = cv2.cvtColor(outputpic, cv2.COLOR_RGB2BGR)
    save_path = os.path.join(path, index).replace(".jpg", ".png")
    cv2.imwrite(save_path, outputpic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='', help='test data root path')
    parser.add_argument('--weights_path', type=str, required=True, help='initial weights path')
    parser.add_argument('--save_path', type=str, default='./results', help='output save image path')

    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()
    main(opt)