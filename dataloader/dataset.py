import os	
from torch.utils.data import DataLoader
from PIL import Image	
import torchvision.transforms as transforms


from dataloader.salient_depth_dataset import SalientDepthDataset
from dataloader.salient_thermal_dataset import SalientThermalDataset

# dataloader for training
def get_loader(image_root, gt_root, ti_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=False, task='RGBT'):
    if task == 'RGBT':
        dataset = SalientThermalDataset(image_root, gt_root, ti_root,  trainsize)
    elif task == 'RGBD':
        dataset = SalientDepthDataset(image_root, gt_root, ti_root,  trainsize)


    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    # print(len(data_loader))
    return data_loader


# test dataset and loader	
class test_dataset_thermal:	
    def __init__(self, image_root, gt_root, ti_root,testsize):	
        self.testsize = testsize	
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
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])	
        self.size = len(self.images)	
        self.index = 0	

    def load_data(self):	
        image = self.rgb_loader(self.images[self.index])	
        gt = self.binary_loader(self.gts[self.index])	
        ti = self.rgb_loader(self.tis[self.index])	
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
    
# test dataset and loader
class test_dataset_depth:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]

        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')]

        self.images = sorted(self.images)
        # print(self.images)
        self.gts = sorted(self.gts)
        # print(self.gts)
        self.depths = sorted(self.depths)
        # print(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.ToTensor()
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
             transforms.ToTensor(),
             transforms.Normalize([0.485], [0.229])
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])
        depth = self.binary_loader(self.depths[self.index])
        # image, gt, depth = self.resize(image, gt, depth)
        image = self.transform(image).unsqueeze(0)
        gt = self.gt_transform(gt).unsqueeze(0)
        depth = self.depths_transform(depth)
        # depth = torch.div(depth.float(), 255.0)  # DUT
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
        # assert img.size == gt.size and gt.size == depth.size
        h = self.testsize
        w = self.testsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
                                                                                                  Image.NEAREST)

    def __len__(self):
        return self.size