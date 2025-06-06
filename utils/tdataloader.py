import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import random

random.seed(2021)


class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, body_root, detail_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.bodys = [body_root + f for f in os.listdir(body_root) if f.endswith('.png')]
        self.details = [detail_root + f for f in os.listdir(detail_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.bodys = sorted(self.bodys)
        self.details = sorted(self.details)
        self.filter_files()
        self.size = len(self.images)
        # self.kernel = np.ones((5, 5), np.uint8)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.ge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def getFlip(self):
        p = random.randint(0, 1)
        self.flip = transforms.RandomHorizontalFlip(p)
        
    def __getitem__(self, index):
        self.getFlip()
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        body = cv2.imread(self.bodys[index],0)
        detail = cv2.imread(self.details[index],0)
        image = self.flip(image)
        image = self.img_transform(image)
        gt = self.flip(gt)
        gt = self.ge_transform(gt)
        body = Image.fromarray(body)
        body = self.flip(body)		
        body = self.ge_transform(body)
        detail = Image.fromarray(detail)
        detail = self.flip(detail)		
        detail = self.ge_transform(detail)
        return image, gt, body, detail

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        bodys = []
        details = []
        for img_path, gt_path, body_path, detail_path in zip(self.images, self.gts, self.bodys, self.details):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            body = Image.open(body_path)
            detail = Image.open(detail_path)
            if img.size == gt.size and img.size == body.size and img.size == detail.size:
                images.append(img_path)
                gts.append(gt_path)
                bodys.append(body_path)
                details.append(detail_path)
        self.images = images
        self.gts = gts
        self.bodys = bodys
        self.details = details

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class test_dataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class test_loader_faster(data.Dataset):
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.size = len(self.images)

    def __getitem__(self, index):
        images = self.rgb_loader(self.images[index])
        images = self.transform(images)

        img_name_list = self.images[index]

        return images, img_name_list

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, body_root, detail_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, body_root, detail_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
