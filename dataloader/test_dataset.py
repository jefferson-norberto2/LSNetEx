import os
from PIL import Image
import torchvision.transforms as transforms

# test dataset and loader
class TestDataset:
    def __init__(self, image_root, gt_root, depth_root, testsize, task='RGBT'):
        self.task = task
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]

        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        if self.task == 'RGBD':
            self.depths_transform = transforms.Compose([
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485], [0.229])
            ])
        else:
             self.depths_transform = transforms.Compose([
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])
        
        if self.task == 'RGBD':
            depth = self.binary_loader(self.depths[self.index])
        else: 
            depth = self.rgb_loader(self.depths[self.index])	

        image = self.transform(image).unsqueeze(0)
        gt = self.gt_transform(gt).unsqueeze(0)
        depth = self.depths_transform(depth)
        depth = depth.unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        h = self.testsize
        w = self.testsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
                                                                                                  Image.NEAREST)

    def __len__(self):
        return self.size

