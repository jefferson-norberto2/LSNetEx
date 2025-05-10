import time
from torch import device, load, cat, no_grad, sigmoid, sum, abs, numel, tensor
from torch.cuda import is_available
from torchvision.utils import make_grid

from lsnetex.dataloader.test_dataset import TestDataset
from lsnetex.models.LSNetEx import LSNetEx
from lsnetex.config import opt, save_name, task
from tensorboardX import SummaryWriter

import wandb

dataset_path = opt.test_path
model_path = opt.model_path

# Verifica se há suporte para GPU
my_device = device("cuda" if is_available() else "cpu")
print('Device in use:', my_device)

wandb.init(
    project="LSNetEx - val", 
        sync_tensorboard=True, 
        name=save_name,
        mode=opt.wandb_mode,
        config={
        "learning_rate": opt.lr,
        "batch_size": opt.batchsize,
        "epochs": opt.epoch,
        "dataset_path": dataset_path,
        "dataset": task,
        })

writer = SummaryWriter(opt.test_save_path + 'summary', flush_secs=30)

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

for index, dataset in enumerate(test_datasets):
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


    tp_sum, fp_sum, fn_sum, tn_sum, iou_sum, dice_sum = 0, 0, 0, 0, 0, 0

    # Calcular tempo de execução
    start_time = time.time()

    for i in range(test_loader.size):
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
            predict_binary = (predict > 0.5).float()

            # MAE
            mae = sum(abs(predict - gt)) / numel(gt)
            mae_sum += mae.item()

            # IoU
            intersection = (predict_binary * gt).sum()
            union = (predict_binary + gt).clamp(0, 1).sum()
            iou_sum += (intersection / (union + 1e-8)).item()

            # Dice Coefficient
            dice_sum += (2 * intersection / (predict_binary.sum() + gt.sum() + 1e-8)).item()

            # True Positives, False Positives, etc.
            tp_sum += intersection.item()
            fp_sum += (predict_binary * (1 - gt)).sum().item()
            fn_sum += ((1 - predict_binary) * gt).sum().item()
            tn_sum += ((1 - predict_binary) * (1 - gt)).sum().item()

            # Salvar imagem predita no wandb
            if i % 100 == 0:
                grid_image = make_grid(image.clone().cpu().data, 1, normalize=True)
                writer.add_image('test/Image', grid_image, i)
                grid_image = make_grid(gt.clone().cpu().data, 1, normalize=True)
                writer.add_image('test/Ground_truth', grid_image, i)
                grid_image = make_grid(ti.clone().cpu().data, 1, normalize=True)
                writer.add_image('test/bound', grid_image, i)

                grid_image = make_grid(predict.clone().cpu().data, 1, normalize=True)
                writer.add_image('OUT/out', grid_image, i, dataformats='CHW')
                grid_image = make_grid(predict_binary.clone().cpu().data, 1, normalize=True)
                writer.add_image('OUT/bound', grid_image, i, dataformats='CHW')
    
    # Calcular tempo total
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for {dataset}: {elapsed_time:.2f} seconds")

    # Cálculos finais
    test_mae = mae_sum / test_loader.size
    test_iou = iou_sum / test_loader.size
    test_dice = dice_sum / test_loader.size
    test_precision = tp_sum / (tp_sum + fp_sum + 1e-8)
    test_recall = tp_sum / (tp_sum + fn_sum + 1e-8)
    test_specificity = tn_sum / (tn_sum + fp_sum + 1e-8)

    writer.add_scalar('MAE', test_mae, global_step=index)
    writer.add_scalar('IoU', test_iou, global_step=index)
    writer.add_scalar('Dice', test_dice, global_step=index)
    writer.add_scalar('Precision', test_precision, global_step=index)
    writer.add_scalar('Recall', test_recall, global_step=index)
    writer.add_scalar('Specificity', test_specificity, global_step=index)
    print(f"MAE: {test_mae}, IoU: {test_iou}, Dice: {test_dice}, Precision: {test_precision}, Recall: {test_recall}, Specificity: {test_specificity}")

print('Test Done!')
writer.close()
wandb.finish()
time.sleep(5)
