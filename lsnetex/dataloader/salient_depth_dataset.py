import torchvision.transforms as transforms

from PIL import Image
from lsnetex.dataloader.salient_dataset import SalientDataset 

# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalientDepthDataset(SalientDataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize):
        super().__init__(image_root, gt_root, depth_root, trainsize)
        self.tis_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        tis = self.binary_loader(self.tis[index])
        
        return self.apply_agumentation(image, gt, tis)

    def resize(self, img, gt, tis):
        assert img.size == gt.size and gt.size == tis.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), tis.resize((w, h),
                                                                                                  Image.NEAREST)
