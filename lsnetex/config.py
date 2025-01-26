from argparse import ArgumentParser
parser = ArgumentParser()

root = '../Datasets'
task = 'RGBT'
network = 0
save_name = f'network_{network}'
save_path = save_name.replace(':', '_')
save_path = save_path.replace(' ', '_')
best_model_path = f'runs/train/{save_name}/Net_epoch_best.pth'

# train/val
parser.add_argument('--task', type=str, default=task, help='type task (RGBT or RGBD)')
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--network', type=int, default=network, help='Choose network encoder: 0 -> V2 Acrticle, 1 -> V3 Small, 2 -> V3 Large, 3 -> V3 Large++')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--train_root', type=str, default=f'{root}/{task}_dataset/train', help='the train images root')
parser.add_argument('--val_root', type=str, default=f'{root}/{task}_dataset/val', help='the val images root')
parser.add_argument('--save_path', type=str, default=f'runs/train/{save_path}/', help='the path to save models and logs')
parser.add_argument('--wandb_mode', type=str, default=f'online', help='type mode (disable, offline or online)')

# test(predict)
parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--test_path',type=str,default=f'{root}/{task}_dataset/test/',help='test dataset path')
parser.add_argument('--test_save_path', type=str, default=f'runs/test/{save_path}', help='path to save run test')
parser.add_argument('--model_path', type=str, default=best_model_path, help='path to model')
opt = parser.parse_args()
