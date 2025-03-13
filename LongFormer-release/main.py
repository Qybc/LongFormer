import argparse
import datetime
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import util.misc as utils
from datasets.adni import ADNIDataset
from models.criterion import Criterion
from models.longformer import Longformer

import torch.nn.functional as F

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score


def get_args_parser():
    parser = argparse.ArgumentParser('Longformer', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr_drop', default=40, type=int, nargs='+')


    # Model parameters
    # * Backbone
    parser.add_argument('--vision_encoder', default='vitg-adapter', type=str,
                        help="vitg-adpater/vit/res50")
    parser.add_argument('--num_feature_scales', default=4, type=int, help='number of feature levels/scales')
    parser.add_argument('--n_times', default=1, type=int)

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', default=288, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--nheads', default=6, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=125, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_classes', default=2, type=int,
                        help="Number of classes")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)


    # * Matcher
    parser.add_argument('--set_cost_loc', default=5, type=float, help="Localization coefficient in the matching cost")
    parser.add_argument('--set_cost_cls', default=2, type=float, help="Classification coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--loc_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='ADNI')
    parser.add_argument('--classification_type', default='NC/AD', help='NC/AD or sMCI/pMCI')
    parser.add_argument('--train_data_path', default='/data2/qiuhui/data/adni')
    parser.add_argument('--test_data_path', default='/data2/qiuhui/data/adni')
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--model', default=None, help='load from checkpoint')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
        
    return parser

def main(args):

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    device = torch.device(f'cuda:{args.local_rank}')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = Longformer(args)
    # criterion = Criterion(args)

    model.to(device)
    # criterion.to(device)
    # model.load_state_dict(torch.load('/data2/qiuhui/code/Longformer/output/checkpoint0030_ADNI.pth', map_location=device), strict=False)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    

    dataset_train = ADNIDataset(args.train_data_path, args=args)
    dataset_val = ADNIDataset(args.test_data_path, args=args)
   
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, sampler=val_sampler)

    
    
    if isinstance(model, nn.Module):
        param_groups = model.parameters()
    else:
        param_groups = model
    optimizer = torch.optim.Adam(
            param_groups,
            lr=5e-4,
            eps=1e-4,
            weight_decay=0.0,
        )
    
    output_dir = Path(args.output_dir)


    for epoch in range(args.epochs):
        print("Start training")
        correct = 0
        num = 0
        model.train()
        # criterion.train()
        start_time = time.time()
        for batch_idx, (img, flow, label, img_indicators, img_idx) in enumerate(train_loader):
            img = img.cuda(non_blocking=True)#.half()
            flow = flow.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            img_indicators = img_indicators.cuda(non_blocking=True)

            outputs = model(img, flow, img_indicators, args)
            bce_loss = nn.BCELoss()
            m = nn.Sigmoid()
            loss = bce_loss(m(outputs),F.one_hot(label,num_classes=2).float())
            pred = outputs.argmax(dim=-1)

            print('img_idx', img_idx)
            print('pred: ', pred)
            print('gt: ', label)
            correct += ((outputs.argmax(dim=-1) == label)+0).sum()
            num+= len(label)
            print('TRAIN epoch: {}/{} iter: {}/{}'.format(epoch, args.epochs, batch_idx,len(train_loader)))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        print('epoch: {}, Accuracy: {}, Correct: {}, Total: {} time: {}\n'.format(epoch, correct / num, correct, num, time.asctime(time.localtime(time.time()))))

        if args.output_dir:
            print("Saving ckpt")
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        

        print("Start validation")
        correct = 0
        num = 0
        y_pred = []
        y_test = []
        model.eval()
        for batch_idx, (img, flow, label, img_indicators, img_idx) in enumerate(val_loader):
            img = img.cuda(non_blocking=True)#.half()
            flow = flow.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            img_indicators = img_indicators.cuda(non_blocking=True)
            with torch.no_grad():
                outputs = model(img, flow, img_indicators, args)
            pred = outputs.argmax(dim=-1)
            print('img_idx', img_idx)
            print('pred: ', pred)
            print('gt: ', label)
            correct += ((pred == label)+0).sum()

            y_pred +=pred.tolist()
            y_test +=label.tolist()

            num+= len(label)
            print('EVAL epoch: {}/{} iter: {}/{}'.format(epoch, args.epochs, batch_idx,len(val_loader)))

        acc = metrics.accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        print('Accuracy: {}, Correct: {}, Total: {} '.format(correct / num, correct, num))
        with open('./{}_eval_out.out'.format(args.dataset_file),'a+') as f:
            f.write('epoch: {}, Accuracy: {}, AUC: {}, Correct: {}, Total: {} time: {}\n'.format(epoch, acc, auc, correct, num, time.asctime(time.localtime(time.time()))))


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('One epoch training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Longformer training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



