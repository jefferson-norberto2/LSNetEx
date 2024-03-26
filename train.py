from os import makedirs, environ
from os.path import exists

from torch import load, cat, sigmoid, tensor, save, no_grad, sum, abs, numel, as_tensor
from torch.optim import Adam
from torch.nn.functional import interpolate
from torch.nn import BCEWithLogitsLoss

from datetime import datetime
from torchvision.utils import make_grid
from models.IOUBCE_without_logits_loss import IOUBCEWithoutLogits_loss
from models.IOUBCE_loss import IOUBCE_loss
from utils import adjust_lr, tesnor_bound

from tensorboardX import SummaryWriter
from logging import basicConfig, info, INFO
from torch.backends import cudnn
from config import opt
from torch.cuda import amp

from models.LSNet import LSNet

# set the device for training
cudnn.benchmark = True
cudnn.enabled = True

environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

# build the model
model = LSNet()

if (opt.load is not None):
    model.load_state_dict(load(opt.load))
    print('load model from ', opt.load)

model.cuda()
params = model.parameters()
optimizer = Adam(params, opt.lr)

# set the path
train_dataset_path = opt.train_root
val_dataset_path = opt.val_root
save_path = opt.save_path

if not exists(save_path):
    makedirs(save_path)

# load data
print('load data...')
if opt.task =='RGBT':
    from dataloader.rgbt_dataset import get_loader, test_dataset
    image_root = train_dataset_path  + '/RGB/'
    ti_root = train_dataset_path  + '/T/'
    gt_root = train_dataset_path  + '/GT/'
    val_image_root = val_dataset_path + '/RGB/'
    val_ti_root = val_dataset_path + '/T/'
    val_gt_root = val_dataset_path + '/GT/'
elif opt.task == 'RGBD':
    from dataloader.rgbd_dataset import get_loader, test_dataset
    image_root = train_dataset_path + '/RGB/'
    ti_root = train_dataset_path + '/depth/'
    gt_root = train_dataset_path + '/GT/'
    val_image_root = val_dataset_path + '/RGB/'
    val_ti_root = val_dataset_path + '/depth/'
    val_gt_root = val_dataset_path + '/GT/'
else:
    raise ValueError(f"Unknown task type {opt.task}")

train_loader = get_loader(image_root, gt_root, ti_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(val_image_root, val_gt_root,val_ti_root, opt.trainsize)
total_step = len(train_loader)

basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
info("Model:")
info(model)

info(save_path + "Train")
info("Config")
info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

CE = BCEWithLogitsLoss().cuda()
IOUBCE = IOUBCE_loss().cuda()

IOUBCEWithoutLogits = IOUBCEWithoutLogits_loss().cuda()

step = 0
writer = SummaryWriter(save_path + 'summary', flush_secs = 30)
best_mae = 1
best_epoch = 0
Sacler = amp.GradScaler()

# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, tis) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            tis = tis.cuda()
            gts = gts.cuda()
            if opt.task == 'RGBD':
                tis = cat((tis, tis, tis), dim=1)

            gts2 = interpolate(gts, (112, 112))
            gts3 = interpolate(gts, (56, 56))

            bound = tesnor_bound(gts, 3).cuda()
            bound2 = interpolate(bound, (112, 112))
            bound3 = interpolate(bound, (56, 56))

            out = model(images, tis)

            loss1 = IOUBCE(out[0], gts)
            loss2 = IOUBCE(out[1], gts2)
            loss3 = IOUBCE(out[2], gts3)

            predict_bound0 = out[0]
            predict_bound1 = out[1]
            predict_bound2 = out[2]
            predict_bound0 = tesnor_bound(sigmoid(predict_bound0), 3)
            predict_bound1 = tesnor_bound(sigmoid(predict_bound1), 3)
            predict_bound2 = tesnor_bound(sigmoid(predict_bound2), 3)
            loss6 = IOUBCEWithoutLogits(predict_bound0, bound)
            loss7 = IOUBCEWithoutLogits(predict_bound1, bound2)
            loss8 = IOUBCEWithoutLogits(predict_bound2, bound3)

            loss_sod = loss1 + loss2 + loss3
            loss_bound =  loss6 + loss7 + loss8
            loss_trans =  out[3]
            loss = loss_sod + loss_bound + loss_trans
            loss.backward()
            optimizer.step()
            step = step + 1
            epoch_step = epoch_step + 1
            loss_all = loss.item() + loss_all
            if i % 10 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, loss_sod: {:.4f},'
                      'loss_bound: {:.4f},loss_trans: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item(),
                             loss_sod.item(),loss_bound.item(), loss_trans.item()))
                info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, loss_sod: {:.4f},'
                              'loss_bound: {:.4f},loss_trans: {:.4f} '.
                             format(epoch, opt.epoch, i, total_step, loss.item(),
                                    loss_sod.item(),loss_bound.item(), loss_trans.item()))
                writer.add_scalar('Loss', loss, global_step=step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train/Ground_truth', grid_image, step)
                grid_image = make_grid(bound[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train/bound', grid_image, step)

                res = out[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('OUT/out', tensor(res), step, dataformats='HW')
                res = predict_bound0[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('OUT/bound', tensor(res), step, dataformats='HW')


        loss_all /= epoch_step
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not exists(save_path):
            makedirs(save_path)
        save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with no_grad():
        mae_sum = 0
        for _ in range(test_loader.size):
            image, gt, ti, _ = test_loader.load_data()
            gt = gt.cuda()
            image = image.cuda()
            ti = ti.cuda()
            if opt.task == 'RGBD':
                tis = cat((tis, tis, tis), dim=1)

            res = model(image, ti)
            res = sigmoid(res)
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_train = sum(abs(res - gt)) * 1.0 / (numel(gt))
            mae_sum = mae_train.item() + mae_sum

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', as_tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))

        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
