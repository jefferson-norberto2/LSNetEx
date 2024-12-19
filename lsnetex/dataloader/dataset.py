from torch.utils.data import DataLoader
from lsnetex.dataloader.salient_depth_dataset import SalientDepthDataset
from lsnetex.dataloader.salient_thermal_dataset import SalientThermalDataset

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