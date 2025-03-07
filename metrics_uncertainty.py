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

from pathlib import Path
import os
import torch
import json
from argparse import ArgumentParser


@torch.no_grad()
def compute_metrics(experiment_path, device='cpu'):
    pred, gt = get_GS(experiment_path, device=device)
    sc_rmse = SparsificationCurves(
        predictions=pred,
        gts=gt,
        error_type = 'rmse', # 'rmse' or 'mae'
    )

    sc_mae = SparsificationCurves(
        predictions=pred,
        gts=gt,
        error_type = 'mae', # 'rmse' or 'mae'
    )
    _, all_basic_results = sc_rmse.get_all(basic_only=True)
    _, all_ause_rmse_results = sc_rmse.get_all(basic_only=False)
    _, all_ause_mae_results = sc_mae.get_all(basic_only=False)
    all_results = {
        'name': all_basic_results['name'],
        'psnr': all_basic_results['psnr'],
        'ssim': all_basic_results['ssim'],
        'lpips': all_basic_results['lpips'],
        'ause_rmse': all_ause_rmse_results['ause'],
        'ause_mae': all_ause_mae_results['ause'],
    }
    mean = {k:v['us'].mean().item() for k,v in all_results.items()}
    return all_results, mean


def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method

                all_result, mean = compute_metrics(method_dir, basic_only=True, device=torch.device("cuda:0"))

                print("  SSIM : {:>12.7f}".format(mean['ssim'], ".5"))
                print("  PSNR : {:>12.7f}".format(mean['psnr'], ".5"))
                print("  LPIPS: {:>12.7f}".format(mean['lpips'], ".5"))
                print("  AUSE RMSE: {:>12.7f}".format(mean['ause_rmse'], ".5"))
                print("  AUSE MAE: {:>12.7f}".format(mean['ause_mae'], ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": mean['ssim'],
                                                        "PSNR": mean['psnr'],
                                                        "LPIPS": mean['lpips'],
                                                        "AUSE RMSE": mean['ause_rmse'],
                                                        "AUSE MAE": mean['ause_mae']})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(all_result['ssim'], all_result['name'])},
                                                        "PSNR": {name: psnr for psnr, name in zip(all_result['psnr'], all_result['name'])},
                                                        "LPIPS": {name: lp for lp, name in zip(all_result['lpips'], all_result['name'])},
                                                        "AUSE RMSE": {name: ause for ause, name in zip(all_result['ause_rmse'], all_result['name'])},
                                                        "AUSE MAE": {name: ause for ause, name in zip(all_result['ause_mae'], all_result['name'])}})
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)