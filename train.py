import sys
import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data.dateloader import CustomDataset
from data.dataloader_VIF import RGBTDataSet
from option import Options
import timm.optim.optim_factory as optim_factory
from losses import fusion_loss
from networks.restormer import init_model
from utils.utils import create_lr_scheduler, tensor2numpy, save_pic
import torchvision.utils as vutils
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import time
import random
import tqdm

# 以下为包装好的 Logger 类的定义
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



def train_one_epoch(model,  optimizer, lr_scheduler, data_loader, device, loss_function, epoch, sample_dir):
    start_time = time.time()
    model.train()

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)

    # optimizer.zero_grad()

    # data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, _, I_B, _, info = data

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)

        loss_dict, I_fused, att_tuple = model(I_A, I_B, step)

        # loss_dict['loss'].backward()
        loss = loss_dict["loss"]

        lr = optimizer.param_groups[0]["lr"]
        if step % 10 == 0:
            print(
                "[train epoch {}-{}] use time: {:.2f} loss: {:.3f}  prompt_loss loss: {:.3f}  lr: {:.6f}".format(
                    epoch, step, time.time()-start_time, loss.item(), loss_dict['prompt_loss'].item(),  lr))

            images = torch.cat((I_A, I_B, I_fused), dim=3)
            vutils.save_image(images, os.path.join(sample_dir, '{}-{}-{}.jpg'.format(epoch, step, info[0]['name'])))


        if not torch.isfinite(loss_dict['loss']):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        lr_scheduler(loss, optimizer, parameters=model.parameters(),
                    update_grad=True)
        # optimizer.step()
        # lr_scheduler.step()
        optimizer.zero_grad()

    print(
        "[train epoch {}] loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  lr: {:.6f}".format(
            epoch, accu_total_loss.item() / (step + 1),
                   accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
                   accu_color_loss.item() / (step + 1), lr))

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lr, loss_function, sample_dir):

    model.eval()
    accu_total_loss = torch.zeros(1).to(device)
    accu_prompt_loss = torch.zeros(1).to(device)

    evalfold_path = os.path.join(sample_dir, 'epoch-' + str(epoch))
    if os.path.exists(evalfold_path) is False:
        os.makedirs(evalfold_path)
    for step, data in enumerate(data_loader):
        if step > 100:
            break
        I_A, _, I_B,_, info = data
        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            # I_A_hq = I_A_hq.to(device)
            # I_B_hq = I_B_hq.to(device)

        loss_dict, I_fused, att_tuple = model(I_A, I_B, step)
        fused_img_Y = tensor2numpy(I_fused)
        save_pic(fused_img_Y, evalfold_path, str(info[0]['name']))
        # loss, loss_ssim = loss_function(I_A, I_B, I_fused, (prompt_x_list, prompt_t_list), step)
        accu_total_loss += loss_dict['loss'].detach()
        accu_prompt_loss += loss_dict['prompt_loss'].detach()

    print("[val epoch {}] loss: {:.3f}  prompt loss: {:.3f} lr: {:.6f}".format(
        epoch, accu_total_loss.item() / (step + 1),
        accu_prompt_loss.item() / (step + 1),
        lr))

    return accu_total_loss.item() / (step + 1)

def save_model(model, optimizer, lr_scheduler, epoch, save_name, use_dp=False):
    if use_dp == True:
        save_file = {"model": model.module.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
    else:
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
    torch.save(save_file, save_name)

def train(args):
    best_val_loss = 1e5
    val_loss = 1e5
    best_epoch = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda')

    # data_loader = CustomDataset(args)
    # print('trainA images = %d' % len(data_loader))

    net = init_model()
    # net = models.AETransModel()
    net.to(device)
    if args.use_dp == True:
        net = torch.nn.DataParallel(net).cuda()
    train_dataset = CustomDataset(args.data_root, phase="train")
    print('train images = %d' % len(train_dataset))
    val_dataset = CustomDataset(args.data_root, phase="val")
    print('validation images = %d' % len(val_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_works,
                                             collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_works,
                                             collate_fn=val_dataset.collate_fn)




    param_groups = optim_factory.add_weight_decay(net, 0.05)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    lr_scheduler = NativeScaler()

    start_epoch = 0
    loss_function = fusion_loss()
    for epoch in range(start_epoch, args.niter + 1):
        train_loss, train_ssim_loss, train_max_loss, train_color_loss, lr = train_one_epoch(
            model=net,
            optimizer=optimizer,
            data_loader=train_loader,
            lr_scheduler=lr_scheduler,
            device=device,
            epoch=epoch,
            loss_function=loss_function,
            sample_dir=args.training_sample
        )
        print("epoch %d, lr: %.3f: train_loss:%.3f, train_ssim_loss:%.3f, train_max_loss:%.3f, train_color_loss:%.3f, "%(
              epoch, lr, train_loss, train_ssim_loss, train_max_loss, train_color_loss))

        if epoch % args.val_every_epoch == 0 or epoch==1:
            # if epoch % args.val_every_epoch == 0 and epoch != 0:
            val_loss = evaluate(model=net,
                                data_loader=val_loader,
                                device=device,
                                epoch=epoch, lr=args.lr,
                                loss_function=loss_function,
                                sample_dir=args.samples)
            save_name = os.path.join(args.checkpoints, str(epoch) + "_epoch.pth")
            save_model(net, optimizer, lr_scheduler, epoch, save_name)

            if val_loss < best_val_loss:
                save_name = os.path.join(args.checkpoints, "best_checkpoint.pth")
                save_model(net, optimizer, lr_scheduler, epoch, save_name)
                best_val_loss = val_loss
                best_epoch = epoch
            print('epoch %d validation ended, the best epoch is %d, best loss is %.3f'% (epoch, best_epoch, best_val_loss))
            print("-----------------------")
            sys.stdout.flush()

    print("-----------------------")
    print('Ended, totally training %d iter, the best epoch is %d, best loss is %.3f' % (args.niter, best_epoch, best_val_loss))
    print("-----------------------")
    print()


if __name__ == '__main__':
    args = Options().parse()
    log = Logger(args.logging)
    sys.stdout = log
    print('------------ Options -------------\n')
    for k, v in sorted(vars(args).items()):
        print('%s: %s\n' % (str(k), str(v)))
    print('-------------- End ----------------\n')
    # fix the seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    train(args)
