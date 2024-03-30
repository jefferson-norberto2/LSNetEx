import torchvision.transforms as transforms

from PIL import Image
from dataloader.salient_dataset import SalientDataset 

# dataset for training
# The current loader is not using the normalized ti maps for training and test. If you use the normalized ti maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalientThermalDataset(SalientDataset):
    def __init__(self, image_root, gt_root, ti_root,  trainsize):
        super().__init__(image_root, gt_root, ti_root, trainsize)
        self.tis_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        ti = self.rgb_loader(self.tis[index])
        
        return self.apply_agumentation(image, gt, ti)

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