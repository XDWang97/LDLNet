import torch
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from datetime import datetime
from net.ldlnet import Net
from utils.tdataloader import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np

file = open("log/LDLNet.txt", "a")
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True
import torch.nn as nn

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, model, optimizer, epoch):
    model.train()
    loss = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        image, gt, body, detail = pack
        image = Variable(image).cuda()
        mask = Variable(gt).cuda()
        body = Variable(body).cuda()
        detail  = Variable(detail).cuda()
        outb1, outd1, out1, outb2, outd2, out2, outb3, outd3, out3= model(image)
        loss_label = structure_loss(out1, mask)
        loss_body = F.binary_cross_entropy_with_logits(outb1, body)
        loss_detail = F.binary_cross_entropy_with_logits(outd1, detail)

        loss_label2 = structure_loss(out2, mask)
        loss_body2 = F.binary_cross_entropy_with_logits(outb2, body)
        loss_detail2 = F.binary_cross_entropy_with_logits(outd2, detail)

        loss_label3 = structure_loss(out3, mask)
        loss_body3 = F.binary_cross_entropy_with_logits(outb3, body)
        loss_detail3 = F.binary_cross_entropy_with_logits(outd3, detail)
        
        loss_all   = (loss_label+loss_body+loss_detail+loss_label2+loss_body2+loss_detail2+loss_label3+loss_body3+loss_detail3)/3

        # ---- backward ----
        loss_all.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # ---- recording loss ----
        loss.update(loss_all.data, opt.batchsize)
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-3: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,loss.avg))
            file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                       '[lateral-3: {:.4f}]\n'.
                       format(datetime.now(), epoch, opt.epoch, i, total_step,loss.avg))

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if ((epoch + 1) % 5 == 0 and epoch > 14) or (epoch + 1) == opt.epoch:
        torch.save(model.state_dict(), save_path + 'DCNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'DCNet-%d.pth' % epoch)
        file.write('[Saving Snapshot:]' + save_path + 'DCNet-%d.pth' % epoch + '\n')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=30, help='epoch number')
    parser.add_argument('--lr', type=float,default=5e-5, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adamw', help='choosing optimizer Adam')
    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=704, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='LDLNet')
    opt = parser.parse_args()

    # ---- build models ----
    model = Net().cuda()

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    body_root = '{}/Body/'.format(opt.train_path)
    detail_root = '{}/Detail/'.format(opt.train_path)
    

    train_loader = get_loader(image_root, gt_root, body_root, detail_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("Start Training")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)

    file.close()
