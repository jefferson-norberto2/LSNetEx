import os
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from dataloader.data_agumentation import * 

# dataset for training
# The current loader is not using the normalized ti maps for training and test. If you use the normalized ti maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, ti_root,  trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        # print(self.images)
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.tis = [ti_root + f for f in os.listdir(ti_root) if f.endswith('.jpg')
                       or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.tis = sorted(self.tis)

        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.tis_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        ti = self.rgb_loader(self.tis[index])
        image, gt, ti = cv_random_flip(image, gt, ti)
        image, gt, ti = randomCrop(image, gt, ti)
        image, gt, ti = randomRotation(image, gt, ti)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        ti = self.tis_transform(ti)
        return image, gt, ti

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.tis)
        images = []
        gts = []
        tis = []
        for img_path, gt_path, ti_path in zip(self.images, self.gts, self.tis):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            ti = Image.open(ti_path)
            if img.size == gt.size and gt.size == ti.size:
                images.append(img_path)
                gts.append(gt_path)
                tis.append(ti_path)
        self.images = images
        self.gts = gts
        self.tis = tis

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, ti):
        assert img.size == gt.size and gt.size == ti.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), ti.resize((w, h),
                                                                                                      Image.NEAREST)
        else:
            return img, gt, ti

    def __len__(self):
        return self.size