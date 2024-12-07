from torch import device, load, cat, no_grad, sigmoid, sum, abs, numel
from torch.cuda import is_available
from tqdm import tqdm

from lsnetex.dataloader.test_dataset import TestDataset
from lsnetex.models.LSNetEx import LSNetEx
from lsnetex.config import opt

dataset_path = opt.test_path
model_path = opt.model_path

# Verifica se há suporte para GPU
my_device = device("cuda" if is_available() else "cpu")
print('Device in use:', my_device)

# Carrega o modelo
model = LSNetEx(network=opt.network).to(my_device)
model.load_state_dict(load(model_path, map_location=my_device))
model.eval()

# Teste
test_mae = []
if opt.task == 'RGBT':
    test_datasets = ['VT821', 'VT1000', 'VT5000']
elif opt.task == 'RGBD':
    test_datasets = ['NJU2K', 'DES', 'LFSD', 'NLPR', 'SIP']
else:
    raise ValueError(f"Unknown task type {opt.task}")

for dataset in test_datasets:
    mae_sum = 0
    save_path = opt.test_save_path
    save_path = dataset + '/'
    print(f'Dataset: {save_path}')

    if opt.task == 'RGBT':
        image_root = dataset_path + dataset + '/RGB/'
        gt_root = dataset_path + dataset + '/GT/'
        ti_root = dataset_path + dataset + '/T/'
        test_loader = TestDataset(image_root, gt_root, ti_root, opt.testsize, task='RGBT')
    elif opt.task == 'RGBD':
        image_root = dataset_path + dataset + '/RGB/'
        gt_root = dataset_path + dataset + '/GT/'
        ti_root = dataset_path + dataset + '/depth/'
        test_loader = TestDataset(image_root, gt_root, ti_root, opt.testsize, task='RGBD')
    else:
        raise ValueError(f"Unknown task type {opt.task}")


    for i in tqdm(range(test_loader.size)):
        image, gt, ti, name = test_loader.load_data()
        gt = gt.to(my_device)
        image = image.to(my_device)
        ti = ti.to(my_device)

        if opt.task == 'RGBD':
            ti = cat((ti, ti, ti), dim=1)

        with no_grad():
            res = model(image, ti)
            predict = sigmoid(res)
            predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
            mae = sum(abs(predict - gt)) / numel(gt)
            mae_sum = mae.item() + mae_sum
            predict = predict.cpu().numpy().squeeze()

    test_mae.append(mae_sum / test_loader.size)

print('Test Done!', 'MAE', test_mae)
