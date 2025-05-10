#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from sparsification_curves import SparsificationCurves, get_GS
import json
import argparse

@torch.no_grad()
def compute_metrics_rmse(experiment_path, basic_only, device):
    pred, gt = get_GS(experiment_path, device=device)
    sc = SparsificationCurves(
        predictions=pred,
        gts=gt,
        # RMSE
        diff_to_error_fn=lambda x: x.square().mean(dim=-3),
        summarizer_fn=lambda x: x.mean(dim=-1).sqrt().mean(),
        # # PSNR
        # diff_to_error_fn=lambda x: x.square().mean(dim=-3),
        # summarizer_fn=lambda x: x.mean(dim=-1).log10().mul(-10).mean(),
        # # MAE
        # diff_to_error_fn=lambda x: x.abs().mean(dim=-3),
        # summarizer_fn=lambda x: x.mean(),
    )
    _, all_results = sc.get_all(basic_only=basic_only)
    all_results.pop('names')
    if basic_only:
        all_results.pop('ssim')
        all_results.pop('lpips')
    mean = {k:v['us'].mean().item() for k,v in all_results.items()}
    return mean

@torch.no_grad()
def compute_metrics_mae(experiment_path, basic_only, device):
    pred, gt = get_GS(experiment_path, device=device)
    sc = SparsificationCurves(
        predictions=pred,
        gts=gt,
        # RMSE
        # diff_to_error_fn=lambda x: x.square().mean(dim=-3),
        # summarizer_fn=lambda x: x.mean(dim=-1).sqrt().mean(),
        # # PSNR
        # diff_to_error_fn=lambda x: x.square().mean(dim=-3),
        # summarizer_fn=lambda x: x.mean(dim=-1).log10().mul(-10).mean(),
        # # MAE
        diff_to_error_fn=lambda x: x.abs().mean(dim=-3),
        summarizer_fn=lambda x: x.mean(),
    )
    _, all_results = sc.get_all(basic_only=basic_only)
    all_results.pop('names')
    if basic_only:
        all_results.pop('ssim')
        all_results.pop('lpips')
    mean = {k:v['us'].mean().item() for k,v in all_results.items()}
    return mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, default='')
    parser.add_argument('--basic_only', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    print(args.experiment_path)
    df_rmse = compute_metrics_rmse(args.experiment_path, args.basic_only, args.device)
    df_mae = compute_metrics_mae(args.experiment_path, args.basic_only, args.device)
    
    results = {}
    assert df_rmse['psnr'] == df_mae['psnr']
    assert df_rmse['ssim'] == df_mae['ssim']
    assert df_rmse['lpips'] == df_mae['lpips']
    results['psnr'] = df_rmse['psnr']
    results['ssim'] = df_rmse['ssim']
    results['lpips'] = df_rmse['lpips']
    results['ause_rmse'] = df_rmse['ause']
    results['ause_mae'] = df_mae['ause']
    print(results)

    with open(args.experiment_path + "/results.json", 'w') as fp:
        json.dump(results, fp, indent=True)