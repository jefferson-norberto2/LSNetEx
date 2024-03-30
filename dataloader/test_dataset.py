import os
from PIL import Image
import torchvision.transforms as transforms

# test dataset and loader
class TestDataset:
    def __init__(self, image_root, gt_root, ti_root,testsize, task='RGBT'):
        self.testsize = testsize
        self.task = task
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.tis = [ti_root + f for f in os.listdir(ti_root) if f.endswith('.jpg')
                       or f.endswith('.png') or f.endswith('.bmp')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.tis = sorted(self.tis)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.tis_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) if self.task == 'RGBT' else transforms.Normalize([0.485], [0.229])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])
        ti = self.rgb_loader(self.tis[self.index]) if self.task == 'RGBT' else self.binary_loader(self.tis[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.gt_transform(gt).unsqueeze(0)
        ti = self.tis_transform(ti).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, ti,name

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

