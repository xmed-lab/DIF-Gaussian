import os
import shutil
import argparse
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.dst_gs import CBCT_dataset_gs
from models.model import DIF_Gaussian
from utils import convert_cuda, load_config
from evaluate import eval_one_epoch



def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    
    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--dst_name', type=str, default='LUNA16')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--cfg_path', type=str, default=None)
    parser.add_argument('--out_res_scale', type=float, default=1.0)
    parser.add_argument('--eval_npoint', type=int, default=100000)

    parser.add_argument('--local-rank', dest='local_rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', default=False)
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--random_views', action='store_true', default=False)
    parser.add_argument('--resume', type=int, default=None)

    args = parser.parse_args()

    if args.dist:
        args.local_rank = int(os.environ["LOCAL_RANK"]) # Make it compatible with different versions of DDP
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)

    cfg = load_config(args.cfg_path)
    if args.local_rank == 0:
        print(args)
        print(cfg)

        # save config
        save_dir = f'./logs/{args.name}'
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(os.path.join(save_dir, 'config.yaml')):
            time_str = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            shutil.copyfile(
                os.path.join(save_dir, 'config.yaml'), 
                os.path.join(save_dir, f'config_{time_str}.yaml')
            )
        shutil.copyfile(args.cfg_path, os.path.join(save_dir, 'config.yaml'))

    # -- initialize training dataset/loader
    train_dst = CBCT_dataset_gs(
        dst_name=args.dst_name,
        cfg=cfg.dataset,
        split='train', 
        num_views=args.num_views, 
        npoint=args.num_points,
        out_res_scale=args.out_res_scale,
        random_views=args.random_views
    )
    train_sampler = None
    if args.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dst)
    train_loader = DataLoader(
        train_dst, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        num_workers=0, # args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # -- initialize evaluation dataset/loader
    eval_loader = DataLoader(
        CBCT_dataset_gs(
            dst_name=args.dst_name,
            cfg=cfg.dataset,
            split='eval',
            num_views=args.num_views,
            out_res_scale=0.5, # low-res for faster evaluation,
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )

    # -- initialize model
    model = DIF_Gaussian(cfg.model)
    if args.resume:
        print(f'resume model from epoch {args.resume}')
        ckpt = torch.load(
            os.path.join(f'./logs/{args.name}/ep_{args.resume}.pth'),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(ckpt)
    
    model = model.cuda()
    if args.dist:
        model = nn.parallel.DistributedDataParallel(
            model, 
            find_unused_parameters=False,
            device_ids=[args.local_rank]
        )
    
    # -- initialize optimizer, lr scheduler, and loss function
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.98, 
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1, 
        gamma=np.power(args.lr_decay, 1 / args.epoch)
    )
    loss_func = nn.MSELoss()

    start_epoch = 0
    if args.resume:
        start_epoch = args.resume + 1
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=args.resume)

    # -- training starts
    for epoch in range(start_epoch, args.epoch + 1):
        if args.dist:
            train_loader.sampler.set_epoch(epoch)

        loss_list = []
        model.train()
        optimizer.zero_grad()

        for k, item in enumerate(train_loader):
            item = convert_cuda(item)

            pred = model(item)
            loss = loss_func(pred['points_pred'], item['points_gt'])
            loss_list.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        if args.local_rank == 0:
            if epoch % 10 == 0:
                loss = np.mean(loss_list)
                print('epoch: {}, loss: {:.4}'.format(epoch, loss))
            
            if epoch % 100 == 0 or (epoch >= (args.epoch - 100) and epoch % 10 == 0):
                if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()
                torch.save(
                    model_state,
                    os.path.join(save_dir, f'ep_{epoch}.pth')
                )

            if epoch % 50 == 0 or (epoch >= (args.epoch - 100) and epoch % 20 == 0):
                metrics, _ = eval_one_epoch(
                    model, 
                    eval_loader, 
                    args.eval_npoint,
                    ignore_msg=True,
                )
                msg = f' --- epoch {epoch}'
                for dst_name in metrics.keys():
                    msg += f', {dst_name}'
                    met = metrics[dst_name]
                    for key, val in met.items():
                        msg += ', {}: {:.4}'.format(key, val)
                print(msg)
        
        if lr_scheduler is not None:
            lr_scheduler.step()
