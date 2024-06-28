import os
import csv
import json
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from datasets.dst_gs import CBCT_dataset_gs
from models.model import DIF_Gaussian
from utils import convert_cuda, load_config, sitk_save



def eval_one_epoch(model, loader, npoint=50000, save_dir=None, ignore_msg=True, use_tqdm=False):
    model.eval()
    results = {}
    metrics = {}
    metrics_tmp = {key:[] for key in ['psnr', 'ssim']}
    if use_tqdm:
        loader = tqdm(loader, ncols=50)
    
    with torch.no_grad():
        for item in loader:
            item = convert_cuda(item)

            dst_name = item['dst_name'][0]
            name = item['name'][0]
            image = item['points_gt'].cpu().numpy()
            image = image[0] # W, H, D

            pred = model(item, is_eval=True, eval_npoint=npoint) # B, 1, N
            output = pred['points_pred']
            output = output[0, 0].data.cpu().numpy()
            
            output = output.reshape(image.shape)
            output = np.clip(output, 0, 1)

            psnr = peak_signal_noise_ratio(image, output, data_range=1.)
            ssim = structural_similarity(image, output, data_range=1.)

            if not ignore_msg:
                print('{}, PSNR: {:.4}, SSIM: {:.4}'.format(
                    name, psnr, ssim
                ))

            dst_res = results.get(dst_name, [])
            dst_met = metrics.get(dst_name, deepcopy(metrics_tmp))

            dst_res.append({
                'name': name, 
                'psnr': psnr,
                'ssim': ssim,
            })
            for key in dst_met.keys():
                dst_met[key].append(dst_res[-1][key])
            
            results[dst_name] = dst_res
            metrics[dst_name] = dst_met

            if save_dir is not None:
                spacing = item['spacing'][0].cpu().numpy()
                origin = item['origin'][0].cpu().numpy()
                save_path = os.path.join(save_dir, f'{name}.nii.gz')
                sitk_save(save_path, output, spacing=spacing, origin=origin, uint8=True)

    for dst_name in metrics.keys():
        dst_met = metrics[dst_name]
        m = {key:np.mean(val) for key, val in dst_met.items()}
        metrics[dst_name] = m

    return metrics, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')

    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--dst_name', type=str, default='LUNA16')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--cfg_path', type=str, default=None)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--view_offset', type=int, default=0)
    parser.add_argument('--out_res_scale', type=float, default=1.0)
    parser.add_argument('--eval_npoint', type=int, default=100000)
    parser.add_argument('--save_results', action='store_true', default=False)

    args = parser.parse_args()
    if args.cfg_path is None:
        args.cfg_path = f'./logs/{args.name}/config.yaml'
    
    print(args)

    cfg = load_config(args.cfg_path)

    # -- dataloader
    eval_loader = DataLoader(
        CBCT_dataset_gs(
            dst_name=args.dst_name,
            cfg=cfg.dataset,
            split=args.split, 
            num_views=args.num_views,
            out_res_scale=args.out_res_scale,
            view_offset=args.view_offset,
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )

    # -- model, load ckpt
    ckpt_path = f'./logs/{args.name}/ep_{args.epoch}.pth'
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    print('load ckpt from', ckpt_path)
    
    model = DIF_Gaussian(cfg.model)
    model.load_state_dict(ckpt)
    model = model.cuda()

    # -- output dir
    tag = '{:.1f}x'.format(args.out_res_scale)
    save_dir = None
    if args.save_results:
        save_dir = f'./logs/{args.name}/results/ep_{args.epoch}/predictions_{tag}'
        os.makedirs(save_dir, exist_ok=True)

    # -- evaluate
    metrics, results = eval_one_epoch(
        model, 
        eval_loader, 
        args.eval_npoint,
        save_dir=save_dir,
        use_tqdm=True,
        ignore_msg=False
    )
    print(metrics)

    # -- save results [csv]
    pred_dir = f'./logs/{args.name}/results/ep_{args.epoch}'
    os.makedirs(pred_dir, exist_ok=True)

    csv_file = open(os.path.join(pred_dir, f'results_{tag}.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['dataset', 'obj_id', 'psnr', 'ssim'])

    for dst_name in results.keys():
        dst_res = results[dst_name]
        for res in dst_res:
            csv_writer.writerow([dst_name, res['name'], res['psnr'], res['ssim']])

        dst_avg = metrics[dst_name]
        csv_writer.writerow([dst_name, 'average', dst_avg['psnr'], dst_avg['ssim']])
    
    csv_file.close()
    
    # -- save config [args]
    with open(os.path.join(pred_dir, 'args.json'), 'w') as f:
        args = vars(args)
        json.dump(args, f, indent=4)
